from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from args import TrainArgs
from data.dataset import Dataset
from data.moldata import MultiMolDatapoint, SingleMolDatapoint


class GCNN(nn.Module):
    def __init__(
        self,
        train_args: TrainArgs,
        atom_feature_vector_length,
        number_of_molecules,
        mol_feature_vector_length: int,
    ):
        super().__init__()

        self.train_args = train_args
        self.number_of_molecules = number_of_molecules

        # Build input layer
        self.input_node_level_nn = nn.Linear(atom_feature_vector_length, train_args.hidden_size, train_args.bias)

        # Build node-level NNs
        if train_args.shared_node_level_nns:
            self.node_level_nn = nn.Linear(train_args.hidden_size, train_args.hidden_size, train_args.bias)
        else:  # separate node-level NNs per depth level
            self.node_level_nns = []
            for depth in range(train_args.depth):
                self.node_level_nns.append(nn.Linear(train_args.hidden_size, train_args.hidden_size, train_args.bias))

        # Build readout layer
        self.readout_hidden_nns = []

        readout_input_size = (
            train_args.hidden_size * number_of_molecules + mol_feature_vector_length * number_of_molecules
        )
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
        def forward_helper(adjacency_matrix: Tensor, atom_feature_matrix: Tensor, mol_features: Tensor) -> Tensor:
            """
            Helper function for
            :param adjacency_matrix:
            :param atom_feature_matrix:
            :return:
            """

            # Input
            # lr_helper = latent representation of helper function
            lr_helper = self.input_node_level_nn(atom_feature_matrix)
            # testing code, delete later
            if torch.isnan(lr_helper).any():
                raise ValueError("Warning: datapoint contains NaN after node-level layer.")

            # Message passing
            for depth in range(self.train_args.depth):
                # Aggregation
                match self.train_args.aggregation_method:
                    case "mean":
                        # lr_helper = (torch.mm(adjacency_matrix, lr_helper).T / torch.sum(adjacency_matrix, dim=1)).T

                        # testing code, delete later
                        a = torch.mm(adjacency_matrix, lr_helper).T
                        b = torch.sum(adjacency_matrix, dim=1)
                        c = (a / b).T

                        if torch.isnan(a).any():
                            raise ValueError(f"Warning: a contains NaN after aggregation. Depth: {depth}")
                        if torch.isnan(b).any():
                            raise ValueError(f"Warning: b contains NaN after aggregation. Depth: {depth}")
                        if torch.isnan(c).any():
                            raise ValueError(f"Warning: a contains NaN after aggregation. Depth: {depth}")

                    case "sum":
                        lr_helper = torch.mm(adjacency_matrix, lr_helper)
                    case x:
                        raise ValueError(f"Aggregation method {x} not implemented.")

                # Update
                lr_helper = self.node_level_nns[depth](lr_helper)
                # testing code, delete later
                if torch.isnan(lr_helper).any():
                    raise ValueError("Warning: datapoint contains NaN after update.")

                # Activation
                lr_helper = self.activation_function(lr_helper)
                # testing code, delete later
                if torch.isnan(lr_helper).any():
                    raise ValueError("Warning: datapoint contains NaN after activation.")

                # Dropout
                if self.train_args.node_level_dropout:
                    lr_helper = self.dropout(lr_helper)
                    # testing code, delete later
                    if torch.isnan(lr_helper).any():
                        raise ValueError("Warning: datapoint contains NaN after dropout.")

            # Readout aggregation. Only sun for now.
            # TODO: Implement several readout aggregation methods
            # TODO: See list at: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html
            lr_helper = torch.sum(lr_helper, dim=0)
            # testing code, delete later
            if torch.isnan(lr_helper).any():
                raise ValueError("Warning: datapoint contains NaN after readout aggregation.")

            # Concatenate molecule features to latent representation
            lr_helper = torch.cat((lr_helper, mol_features))
            # testing code, delete later
            if torch.isnan(lr_helper).any():
                raise ValueError("Warning: datapoint contains NaN after concatenation.")

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
        # testing code, delete later
        if torch.isnan(latent_representation).any():
            raise ValueError("Warning: datapoint contains NaN after readout input layer.")

        for depth in range(self.train_args.readout_num_hidden_layers):
            latent_representation = self.readout_hidden_nns[depth](latent_representation)
            # testing code, delete later
            if torch.isnan(latent_representation).any():
                raise ValueError(f"Warning: datapoint contains NaN after readout hidden layer {depth}.")

            # TODO: Using readout dropout leads to NN outputting NaN, fix this
            # if self.train_args.readout_dropout:
            #     latent_representation = self.dropout(latent_representation)

        output = self.readout_output_layer(latent_representation)
        # testing code, delete later
        if torch.isnan(output).any():
            raise ValueError("Warning: datapoint contains NaN after readout output later.")

        return output
