import torch.nn as nn
from args import ModelArgs
from data.dataset import Dataset


class GCNN(nn.Module):
    def __init__(
        self,
        model_args: ModelArgs,
        atom_feature_vector_length,
        mol_feature_vector_length: int,
    ):
        super().__init__()

        self.model_args = model_args

        # Build input layer
        self.input_node_level_nn = nn.Linear(atom_feature_vector_length, model_args.hidden_size, model_args.bias)

        # Build node-level NNs
        if model_args.shared_node_level_nns:
            self.node_level_nn = nn.Linear(model_args.hidden_size, model_args.hidden_size, model_args.bias)
        else:  # separate node-level NNs per depth level
            self.node_level_nns = []
            for depth in range(model_args.depth):
                self.node_level_nns.append(nn.Linear(model_args.hidden_size, model_args.hidden_size, model_args.bias))

        # Build readout layer
        self.readout_hidden_nns = []
        for depth in range(model_args.readout_num_hidden_layers):
            self.readout_hidden_nns.append(nn.Linear(model_args.hidden_size, model_args.hidden_size, model_args.bias))

        self.readout_output_nn = nn.Linear(model_args.hidden_size, 1, model_args.bias)

        # Set dropout layer if using
        if model_args.node_level_dropout or model_args.readout_dropout:
            if model_args.dropout_probability != 0.0:
                self.dropout = nn.Dropout(model_args.readout_dropout)
            else:
                self.dropout = nn.Dropout()

        # Set activation function
        match self.activation:
            case "ReLU":
                self.activation_function = nn.ReLU()
            case _:
                raise ValueError(f"{self.activation}")
