from moldata import SingleMolDatapoint, MultiMolDatapoint
from args import DataArgs
from typing import Union
from data.featurizer import MoleculeFeaturizer
import pandas as pd
from atom_descriptors import get_features_vector_length, get_features_to_normalize


class Dataset:
    def __init__(
        self,
        data_args: DataArgs,
        datapoints: list[Union[SingleMolDatapoint, MultiMolDatapoint]],
        atom_feature_vector_length: int,
        mol_feature_vector_length: int,
        features_to_normalize: list[bool],
    ):
        self.data_args = data_args
        self.datapoints = datapoints
        self.atom_feature_vector_length = atom_feature_vector_length
        self.mol_feature_vector_length = mol_feature_vector_length
        self.features_to_normalize = features_to_normalize

    def normalize_features(self, scaler):
        pass

    def normalize_targets(self, scaler):
        pass


def load_dataset(data_args: DataArgs) -> Dataset:
    featurizer = MoleculeFeaturizer(data_args)
    data = pd.read_csv(data_args.dataset_path)

    datapoints = []
    for _, row in data.iterrows():
        datapoints.extend(
            [
                featurizer.featurize_datapoint(row[c], data_args.number_of_molecules, row[data_args.target_column])
                for c in data_args.molecule_smiles_columns
            ]
        )

    return Dataset(
        data_args,
        datapoints,
        get_features_vector_length(data_args.atom_descriptors),
        0,
        get_features_to_normalize(data_args.atom_descriptors),
    )
