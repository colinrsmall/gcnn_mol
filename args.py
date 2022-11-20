from typing import Literal

from tap import Tap

from data.atom_descriptors import all_descriptors


class ModelArgs(Tap):
    """
    Holds settings and parameters that define the GCNN model.
    """

    depth: int = 3
    """Number of message passing steps during a GCNN forward pass."""

    hidden_size: int = 300
    """Hidden layer size of node-level NNs."""

    activation_function: Literal["ReLU"] = "ReLU"
    """Activation function for node-level NNs and readout FCNN."""

    readout_depth: int = 3
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

    aggregation_method: Literal["mean", "sum"] = "mean"
    """Aggregation method during graph convolution."""

    def process_args(self) -> None:
        """
        Checks that argument choices are valid.
        """
        if not (self.node_level_dropout and self.readout_dropout) and self.dropout_probability != 0.0:
            raise ValueError(
                """User specified dropout probability, but dropout is disabled for both node-level NNs and
                              the final readout FCNN."""
            )


class DataArgs(Tap):
    """
    Holds settings and parameters for data loading and featurization.
    """

    explicit_hydrogens: bool = False
    """Adds explicit hydrogens to the molecule being featurized if explicit hydrogens are not included in the molecule's
           SMILES string"""

    atom_descriptors: list[str] = ["all"]
    """Determines which atom descriptors will be used when featurizing a molecule."""

    dataset_path: str
    """Points to the dataset to be loaded."""

    number_of_molecules: int
    """The number of molecules per datapoint"""

    molecule_smiles_columns: list[str]
    """The names of the columns in the dataset that hold each molecules' SMILES string."""

    target_column: str
    """The name of the column that contains the training target."""

    def process_args(self) -> None:
        """
        Checks that argument choices are valid.
        """
        super(DataArgs, self).process_args()

        # Ensure number of molecules matches the number of molecule smiles columns
        if self.number_of_molecules != len(self.molecule_smiles_columns):
            raise ValueError(
                f"Number of molecules ({self.number_of_molecules}) does not match the number of molecule"
                f"SMILES columns ({len(self.molecule_smiles_columns)})."
            )

        # Ensure chosen descriptors are valid, or load all descriptors if using all
        if self.atom_descriptors != ["all"]:
            for descriptor in self.atom_descriptors:
                if descriptor not in all_descriptors():
                    raise ValueError(f"{descriptor} is not a valid atom descriptor. Please check for typos.")
        else:
            self.atom_descriptors = all_descriptors().keys()
