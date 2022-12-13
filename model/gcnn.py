from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from utils.args import TrainArgs
from data.moldata import MultiMolDatapoint, SingleMolDatapoint

from model.laf_layer import LAFLayerFast


class FCNNOnly(nn.Module):
    def __init__(self, train_args: TrainArgs, mol_feature_vector_length: int, number_of_molecules: int):
        super().__init__()

        self.train_args = train_args
        self.mol_feature_vector_length = mol_feature_vector_length
        self.number_of_molecules = number_of_molecules

        # Build readout layer
        self.readout_hidden_nns = []

        readout_input_size = mol_feature_vector_length * number_of_molecules
        self.readout_input_layer = nn.Linear(readout_input_size, train_args.readout_hidden_size, train_args.bias)

        for depth in range(train_args.readout_num_hidden_layers):
            self.readout_hidden_nns.append(
                nn.Linear(train_args.readout_hidden_size, train_args.readout_hidden_size, train_args.bias)
            )

        self.readout_output_layer = nn.Linear(train_args.readout_hidden_size, 1, train_args.bias)

        # Set dropout layer if using
        if train_args.node_level_dropout or train_args.readout_dropout:
            if train_args.dropout_probability != 0.0:
                self.dropout = nn.Dropout(train_args.dropout_probability)
            else:
                self.dropout = nn.Dropout()

        # Set activation function
        match train_args.activation_function:
            case "ReLU":
                self.activation_function = nn.ReLU()
            case _:
                raise ValueError(f"{self.activation}")

    def forward(self, datapoint: Union[SingleMolDatapoint, MultiMolDatapoint]):
        # Load and concat molecule features, if needed
        if self.number_of_molecules == 1:
            mol_features = datapoint.molecule_features_vector
        else:
            mol_features = torch.concat(datapoint.molecule_feature_vectors)

        # FNN
        latent_representation = self.readout_input_layer(mol_features)

        for depth in range(self.train_args.depth):
            latent_representation = self.readout_hidden_nns[depth](latent_representation)

        output = self.readout_output_layer(latent_representation)

        return output


class GCNN(nn.Module):
    def __init__(
        self,
        train_args: TrainArgs,
        atom_feature_vector_length,
        number_of_molecules,
        mol_feature_vector_length: int,
        device: torch.device,
    ):
        super().__init__()

        self.train_args = train_args
        self.number_of_molecules = number_of_molecules

        # Build graph attention mechanism
        if train_args.graph_attention:
            self.graph_attention = nn.Linear(train_args.hidden_size * 2, 1, train_args.bias, device=device)
            match train_args.graph_attention_activation:
                case "softmax":
                    self.attention_activation = nn.Softmax(dim=1)
                case "sigmoid":
                    self.attention_activation = nn.Sigmoid()
                case x:
                    raise ValueError(f"Graph attention activation {x} not implemented.")

        # Build input layer
        self.input_node_level_nn = nn.Linear(
            atom_feature_vector_length, train_args.hidden_size, train_args.bias, device=device
        )

        # Build node-level NNs
        if train_args.shared_node_level_nns:
            self.node_level_nn = nn.Linear(
                train_args.hidden_size, train_args.hidden_size, train_args.bias, device=device
            )
        else:  # separate node-level NNs per depth level
            self.node_level_nns = []
            for depth in range(train_args.depth):
                self.node_level_nns.append(
                    nn.Linear(train_args.hidden_size, train_args.hidden_size, train_args.bias, device=device)
                )

        # Build LAF aggregation layer if using
        if train_args.aggregation_method == "LAF":
            self.laf = LAFLayerFast(units=1, device=device)
            self.readout_laf = LAFLayerFast(units=1, device=device)

        # Build readout layer
        self.readout_hidden_nns = []

        readout_input_size = (
            train_args.hidden_size * number_of_molecules + mol_feature_vector_length * number_of_molecules
        )
        self.readout_input_layer = nn.Linear(
            readout_input_size, train_args.readout_hidden_size, train_args.bias, device=device
        )

        for depth in range(train_args.readout_num_hidden_layers):
            self.readout_hidden_nns.append(
                nn.Linear(
                    train_args.readout_hidden_size, train_args.readout_hidden_size, train_args.bias, device=device
                )
            )

        self.readout_output_layer = nn.Linear(train_args.readout_hidden_size, 1, train_args.bias, device=device)

        # Set dropout layer if using
        if train_args.node_level_dropout or train_args.readout_dropout:
            if train_args.dropout_probability != 0.0:
                self.dropout = nn.Dropout(train_args.dropout_probability)
            else:
                self.dropout = nn.Dropout()

        # Set activation function
        match train_args.activation_function:
            case "ReLU":
                self.activation_function = nn.ReLU()
            case "PReLU":
                self.activation_function = nn.PReLU()
            case "SeLU":
                self.activation_function = nn.SELU()
            case "tanh":
                self.activation_function = nn.Tanh()
            case "leakyReLU":
                self.activation_function = nn.LeakyReLU()
            case _:
                raise ValueError(f"{self.activation} not implemented.")

    def forward(self, datapoint: Union[SingleMolDatapoint, MultiMolDatapoint]):
        def forward_helper(adjacency_matrix: Tensor, atom_feature_matrix: Tensor, mol_features: Tensor) -> Tensor:
            """
            Helper function for a forward pass of the NN.
            :param mol_features:
            :param adjacency_matrix:
            :param atom_feature_matrix:
            :return:
            """

            # Input
            # lr_helper = latent representation of helper function
            lr_helper = self.input_node_level_nn(atom_feature_matrix)

            # Message passing
            for depth in range(self.train_args.depth):
                # Graph attention
                if self.train_args.graph_attention:
                    # Create copy of adjacency matrix that will be scaled via attention
                    attentive_adjacency_matrix = torch.clone(adjacency_matrix)
                    # Calculate attention for each atom
                    for atom_idx in range(len(adjacency_matrix)):
                        # Get indices in adjacency matrix of the current atom's neighbors
                        neighbor_indices = torch.where(adjacency_matrix[atom_idx] > 0)
                        neighbors = lr_helper[neighbor_indices]
                        # Run a batch of neighbor-centroid atom pairs through the attention layer
                        attention_vector = self.graph_attention(
                            torch.concat([neighbors, torch.stack([lr_helper[atom_idx]] * len(neighbors))], dim=1)
                        )
                        attention_vector = self.attention_activation(attention_vector)
                        # Scale the adjacency matrix by the calculated attentions
                        attentive_adjacency_matrix[atom_idx, neighbor_indices[0]] = attentive_adjacency_matrix[
                            atom_idx, neighbor_indices[0]
                        ] * attention_vector.view(-1)
                else:
                    attentive_adjacency_matrix = adjacency_matrix

                # Aggregation
                match self.train_args.aggregation_method:
                    case "mean":
                        lr_helper = (
                            torch.mm(attentive_adjacency_matrix, lr_helper).T
                            / torch.sum(attentive_adjacency_matrix, dim=1)
                        ).T
                    case "sum":
                        lr_helper = torch.mm(attentive_adjacency_matrix, lr_helper)
                    case "LAF":
                        lr_helper_after_aggregation = torch.zeros(lr_helper.shape, device=lr_helper.device)
                        # Iterate through each atom_idx in a molecule
                        for atom_idx in range(len(attentive_adjacency_matrix)):
                            # Collect the neighbors (and their features) of the given atom
                            neighbors = lr_helper[torch.where(attentive_adjacency_matrix[atom_idx] > 0)]
                            # Pass the neighbors (and their features) through the LAF layer
                            agg = self.laf(neighbors.T)
                            # Build a separate latent representation so we that don't adjust features mid-aggregation
                            lr_helper_after_aggregation[atom_idx, :] = agg[0, :, 0]
                        lr_helper = lr_helper_after_aggregation
                    case x:
                        raise ValueError(f"Aggregation method {x} not implemented.")

                # Update
                lr_helper = self.node_level_nns[depth](lr_helper)

                # Activation
                lr_helper = self.activation_function(lr_helper)

                # Dropout
                if self.train_args.node_level_dropout:
                    lr_helper = self.dropout(lr_helper)

            # Readout aggregation
            match self.train_args.aggregation_method:
                case "sum":
                    lr_helper = torch.sum(lr_helper, dim=0)
                case "LAF":
                    lr_helper = self.readout_laf(lr_helper.T)[0, :, 0]
                case x:
                    raise ValueError(f"Aggregation method {x} not implemented.")

            # Concatenate molecule features to latent representation
            lr_helper = torch.cat((lr_helper, mol_features))

            return lr_helper

        # Use helper function to perform message passing step(s)
        if self.number_of_molecules == 1:
            latent_representation = forward_helper(
                datapoint.adjacency_matrix, datapoint.atom_feature_matrix, datapoint.molecule_features_vector
            )
        else:
            latent_representations = []
            for i in range(self.number_of_molecules):
                lr = forward_helper(
                    datapoint.adjacency_matrices[i],
                    datapoint.atom_feature_matrices[i],
                    datapoint.molecule_feature_vectors[i],
                )
                latent_representations.append(lr)
            latent_representation = torch.concat(latent_representations)

        # Readout
        latent_representation = self.readout_input_layer(latent_representation)
        latent_representation = self.activation_function(latent_representation)

        for depth in range(self.train_args.readout_num_hidden_layers):
            latent_representation = self.readout_hidden_nns[depth](latent_representation)

            if self.train_args.readout_dropout:
                latent_representation = self.dropout(latent_representation)

            latent_representation = self.activation_function(latent_representation)

        output = self.readout_output_layer(latent_representation)

        return output
