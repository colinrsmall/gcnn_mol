from datetime import datetime
from typing import Literal

from tap import Tap

from data import atom_descriptors, molecule_descriptors

from . import metrics


class TrainArgs(Tap):
    """
    Holds settings and parameters required for training a model.
    """

    # ~~~ Model Arguments ~~~

    depth: int = 3
    """Number of message passing steps during a GCNN forward pass. Also defines number of hidden layers."""

    hidden_size: int = 300
    """Hidden layer size of node-level NNs."""

    activation_function: Literal["ReLU"] = "ReLU"
    """Activation function for node-level NNs and readout FCNN."""

    readout_num_hidden_layers: int = 3
    """Number of layers in the readout FCNN."""

    readout_hidden_size: int = None
    """Hidden layer size for the readout FCNN. Defaults to hidden_size."""

    bias: bool = True
    """Whether to enable bias in node-level NNs and in readout NN."""

    node_level_dropout: bool = False
    """Whether to enable dropout in node-level NNs."""

    readout_dropout: bool = False
    """Whether to enable dropout in readout FCNN."""

    dropout_probability: float = 0.0
    """Dropout probability."""

    aggregation_method: Literal["mean", "sum", "LAF"] = "LAF"
    """Aggregation method during graph convolution."""

    shared_node_level_nns: bool = False
    """Whether node-level NNs should be shared across the depth of message passing or not."""

    mol_features_only: bool = False
    """Whether the model should use molecule features only. If true, the model will not use graph convolution and will
       only feed molecule features through the readout fully connected NN."""

    # ~~~ Data Arguments ~~~

    explicit_hydrogens: bool = False
    """Adds explicit hydrogens to the molecule being featurized if explicit hydrogens are not included in the molecule's
           SMILES string"""

    atom_descriptors: list[str] = ["all"]
    """Determines which atom descriptors will be used when featurizing a molecule."""

    molecule_descriptors: list[str] = ["all"]
    """Determines which molecule descriptors will be used when featurizing a molecule."""

    dataset_path: str
    """Points to the dataset to be loaded."""

    number_of_molecules: int
    """The number of molecules per datapoint"""

    molecule_smiles_columns: list[str]
    """The names of the columns in the dataset that hold each molecules' SMILES string."""

    target_column: str
    """The name of the column that contains the training target."""

    # ~~~ Misc. Training Args ~~~

    model_save_path: str = None
    """Path at which to save the trained model and data scalers. If None, model will not be saved."""

    model_save_name: str = None
    """The name of the model to save at the given model save path. If None, defaults to the datetime stamp when the
       model's training process is started."""

    epochs: int = 100
    """The number of epochs to train the model for."""

    metrics: list[str] = ["mse"]
    """Which metrics to evaluate the model's performance with."""

    model_save_metric: str = metrics[0]
    """The metric with which the best performing models are saved during model training. Defaults to the first metric
       passed in the list of metrics to evaluate."""

    log_file_path: str = None
    """The path to save the model training log file at. If None, no log file will be generated."""

    wandb_logging: bool = False
    """If passed, model training will be logged on wandb."""

    sgd_nesterov: bool = False
    """Enables Nesterov momentum for SGD optimizer."""

    sgd_lr: float = 0.001
    """Learning rate for SGD optimizer."""

    sgd_momentum: float = 0.9
    """Momentum factor for SGD optimizer."""

    sgd_weight_decay: float = 0
    """Weight decay (L2 penalty) for SGD optimizer."""

    sgd_dampening: float = 0
    """Dampening rate for momentum for SGD optimizer."""

    loss_function: Literal["mse", "mae"] = "mae"
    """Loss function used when training the model."""

    def process_args(self) -> None:
        """
        Checks that argument choices are valid.
        """
        super(TrainArgs, self).process_args()

        # Make sure dropout probability isn't erroneously set
        if not (self.node_level_dropout and self.readout_dropout) and self.dropout_probability != 0.0:
            raise ValueError(
                """User specified dropout probability, but dropout is disabled for both node-level NNs and
                              the final readout FCNN."""
            )

        # Set readout hidden size if not dps
        if not self.readout_hidden_size:
            self.readout_hidden_size = self.hidden_size

        # Ensure number of molecules matches the number of molecule smiles columns
        if self.number_of_molecules != len(self.molecule_smiles_columns):
            raise ValueError(
                f"Number of molecules ({self.number_of_molecules}) does not match the number of molecule"
                f"SMILES columns ({len(self.molecule_smiles_columns)})."
            )

        # Ensure chosen atom descriptors are valid, or load all descriptors if using all
        if self.atom_descriptors != ["all"]:
            for descriptor in self.atom_descriptors:
                if descriptor not in atom_descriptors.all_descriptors():
                    raise ValueError(f"{descriptor} is not a valid atom descriptor. Please check for typos.")
        else:
            self.atom_descriptors = atom_descriptors.all_descriptors().keys()

        # Ensure chosen molecule descriptors are valid, or load all descriptors if using all
        if self.molecule_descriptors != ["all"]:
            for descriptor in self.molecule_descriptors:
                if descriptor not in molecule_descriptors.all_descriptors():
                    raise ValueError(f"{descriptor} is not a valid atom descriptor. Please check for typos.")
        else:
            self.molecule_descriptors = molecule_descriptors.all_descriptors().keys()

        # Ensure chosen metrics are valid
        for metric in self.metrics:
            if metric not in metrics.all_metrics():
                raise ValueError(
                    f"Metric {metric} is not a valid metric. Check utils/metrics.py for list of valid metrics."
                )

        # Set model save name to the current datetime stamp if no model name is provided
        if self.model_save_name is None:
            self.model_save_name = f"gcnn_mol_trained_{str(datetime.now())}"