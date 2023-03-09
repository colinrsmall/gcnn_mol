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

    activation_function: Literal["ReLU", "PReLU", "SeLU", "leakyReLU"] = "ReLU"
    """Activation function for node-level NNs and readout FCNN."""

    readout_num_hidden_layers: int = 3
    """Number of layers in the readout FCNN."""

    readout_hidden_size: int = None
    """Hidden layer size for the readout FCNN. Defaults to hidden_size."""

    bias: bool = True
    """Whether to enable bias in node-level NNs and in readout NN."""

    node_level_dropout: bool = True
    """Whether to enable dropout in node-level NNs."""

    readout_dropout: bool = True
    """Whether to enable dropout in readout FCNN."""

    dropout_probability: float = 0.0
    """Dropout probability."""

    aggregation_method: Literal["mean", "sum", "LAF"] = "sum"
    """Aggregation method during graph convolution."""

    shared_node_level_nns: bool = False
    """Whether node-level atom NNs should be shared across the depth of message passing or not."""

    bond_weighting: bool = False
    """If passed, bond types are used to weight neighboring atoms during aggregation."""

    attention_uses_atom_features: bool = False
    """If passed, the attention mechanism uses only the atom features as input, not the current latent represenation."""

    graph_attention: bool = False
    """Whether to enable graph attention layer."""

    graph_attention_activation: Literal["sigmoid", "softmax"] = "sigmoid"
    """Which activation function to use on the output of the graph attention layer. Velickovic et al. use SoftMax."""

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

    molecule_smiles_columns: str
    """The names of the columns in the dataset that hold each molecules' SMILES string."""

    target_column: str
    """The name of the column that contains the training target."""

    # ~~~ Misc. Training Args ~~~

    seed: int = None
    """Seed used for torch and sklearn."""

    model_save_path: str = None
    """Path at which to save the trained model and data scalers. If None, model will not be saved."""

    model_save_name: str = None
    """The name of the model to save at the given model save path. If None, defaults to the datetime stamp when the
       model's training process is started."""

    epochs: int = 100
    """The number of epochs to train the model for."""

    metrics: list[str] = ["mse", "rmse", "mae", "kendall", "spearman"]
    """Which metrics to evaluate the model's performance with."""

    model_save_metric: str = metrics[0]
    """The metric with which the best performing models are saved during model training. Defaults to the first metric
       passed in the list of metrics to evaluate."""

    log_file_path: str = None
    """The path to save the model training log file at. If None, no log file will be generated."""

    wandb_logging: bool = False
    """If passed, model training will be logged on wandb."""

    learning_rate: float = 0.0003
    """Learning rate for Adam optimizer."""

    loss_function: Literal["mse", "mae"] = "mse"
    """Loss function used when training the model."""

    cpu: bool = False
    """Force training to use the CPU instead of an available GPU."""

    detect_anomalies: bool = False
    """Whether to set torch.autograd.set_detect_anomaly(True) when training."""

    sweep_config: str = None
    """If provided, arguments in the pointed to YAML file will be used as part of a wandb hyperparameter swep."""

    gradient_clipping_norm: int = 1
    """Gradient clipping cutoff value."""

    optimizer: Literal["adam", "sgd", "adagrad", "adadelta", "sgd_nesterov", "adamw"] = "sgd"
    """Which optimizer to use while training the model."""

    co_attention_legacy: bool = False
    """If passed, model uses co-attention with multi-molecule datapoints. Uses a legacy version of co-attention that simply adds connections between the nodes."""

    co_attention: bool = False
    """If passed, model uses co-attention with multi-molecule datapoints."""

    update_before_aggregation: bool = False
    """If passed, the feature vectors/latent representations of the molecules are updated before aggregation."""

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

        # Parse and split molecule smiles columns
        self.molecule_smiles_columns = self.molecule_smiles_columns.split(",")

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
        if self.metrics == ["all"]:  # Workaround for wandb hyperparameter sweep
            self.metrics = ["mse", "rmse", "mae", "kendall", "spearman", "pearson"]
        for metric in self.metrics:
            if metric not in metrics.all_metrics():
                raise ValueError(
                    f"Metric {metric} is not a valid metric. Check utils/metrics.py for list of valid metrics."
                )

        # Set model save name to the current datetime stamp if no model name is provided
        if self.model_save_name is None:
            self.model_save_name = f"gcnn_mol_trained_{str(datetime.now())}"

        # Raise error if user tries to use co-attention without using a multi-molecule dataset
        if self.number_of_molecules == 1 and self.co_attention_legacy:
            raise ValueError("Co-attention can only be used with datasets with multi-molecule datapoints.")
