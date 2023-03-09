from typing import Union

import pandas as pd
import torch
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils import data
from tqdm.auto import tqdm

from utils.args import TrainArgs
from data.atom_descriptors import get_features_to_normalize
from data.featurizer import MoleculeFeaturizer
from data.moldata import AbstractDatapoint, MultiMolDatapoint, SingleMolDatapoint

from rdkit.Chem import rdPartialCharges


class Dataset(data.Dataset):
    # TODO: Inherit from nn.Dataset (https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
    def __init__(
        self,
        train_args: TrainArgs,
        datapoints: list[Union[SingleMolDatapoint, MultiMolDatapoint]],
        features_to_normalize: list[bool],
        num_molecules_per_datapoint: int,
    ):
        self.data_args = train_args
        self.datapoints = datapoints
        self.features_to_normalize = features_to_normalize
        self.num_molecules_per_datapoint = num_molecules_per_datapoint

        # Compute atom and mol feature vectors length by featurizing dummy atom and mol
        featurizer = MoleculeFeaturizer(train_args)
        mol = Chem.MolFromSmiles("CC")
        rdPartialCharges.ComputeGasteigerCharges(mol)
        self.atom_features_vector_length = len(featurizer.create_descriptors_for_atom(mol.GetAtoms()[0]))
        self.mol_features_vector_length = len(featurizer.create_descriptors_for_molecule(mol))

    def fit_scalers_to_features(self) -> list[StandardScaler]:
        """
        Fit a set of scalers to the features of the dataset.
        :return: A list that matches the length of the feature vectors generated for each atom. Contains None at indices
        corresponding to features that should not be scaled or an sklearn StandardScaler for features that should be.
        """

        # Initialize scaler and stack each molecule datapoints' atom feature matrices into a single matrix
        scalers = [(StandardScaler() if f else None) for f in self.features_to_normalize]

        if self.num_molecules_per_datapoint > 1:
            atom_features_stack = torch.vstack([torch.vstack(dp.atom_feature_matrices) for dp in self.datapoints])
        else:
            atom_features_stack = torch.vstack([dp.atom_feature_matrix for dp in self.datapoints])

        # Fit each scaler
        for i, feature in enumerate(self.features_to_normalize):
            if feature:
                feature_vector = atom_features_stack[:, i]
                scalers[i].fit(feature_vector.reshape([-1, 1]))

        return scalers

    def normalize_features(self, scalers: list[StandardScaler]) -> None:
        """
        Normalize the features of the dataset in place using a provided list of scalers.
        :param scalers: A list of sklearn StandardScalers. Length must be equal to the number of atom features in the
        dataset.
        """

        def _feature_helper(feature_matrix: torch.Tensor, scalers: list[StandardScaler]):
            for i, scaler in enumerate(scalers):
                if scaler:  # Multiple reshapes are required because sklearn.StandardScaler expects input shape of
                    feature_matrix[:, i] = torch.from_numpy(  # [-1, 1] but the feature matrix column is [-1]
                        scaler.transform(feature_matrix[:, i].reshape([-1, 1])).reshape([-1])
                    )

        # Ensure the list of scalers matches the number of features to scale
        if len(scalers) != len(self.features_to_normalize):
            raise ValueError("Mismatch between the number of scalers and the number of features to scale.")

        for dp in self.datapoints:
            if self.num_molecules_per_datapoint > 1:
                for mol_feature_matrix in dp.atom_feature_matrices:
                    _feature_helper(mol_feature_matrix, scalers)
            else:
                _feature_helper(dp.atom_feature_matrix, scalers)

    def to(self, device: torch.device) -> None:
        """
        Send the tensors and targets contained in the dataset to a provided device.
        :param device: A torch.device to send the tensors to.
        """
        for dp in self.datapoints:
            dp.to(device)

    def create_paired_datapoints(self, co_attention_factor: float) -> None:
        for dp in self.datapoints:
            dp.create_paired_datapoints(co_attention_factor)

    def __len__(self) -> int:
        return len(self.datapoints)

    def __getitem__(self, idx) -> AbstractDatapoint:
        return self.datapoints[idx]

    def train_test_split(self, test_split_percentage) -> (list[torch.Tensor], list[torch.Tensor]):
        """
        Splits the dataset into two parts, a training split of size len(dataset)*(1-test_split_percentage) and a
        test split of size len(dataset)*(test_split_percentage). Randomly assigns datapoints to each split.
        :param test_split_percentage: The size of the test split as a percentage of the total size of the dataset.
        :return: Two lists of datapoints corresponding to the training split and test split.
        """
        if self.data_args.seed:
            train_set, test_set = train_test_split(
                self.datapoints, test_size=test_split_percentage, random_state=self.data_args.seed
            )
        else:
            train_set, test_set = train_test_split(self.datapoints, test_size=test_split_percentage)
        return train_set, test_set


def load_dataset(train_args: TrainArgs) -> Dataset:
    featurizer = MoleculeFeaturizer(train_args)
    data = pd.read_csv(train_args.dataset_path)

    datapoints = []
    print("Loading and featurizing molecules:")

    bad_smiles_count = 0
    for _, row in tqdm(data.iterrows(), total=len(list(data.iterrows()))):
        try:
            dp = featurizer.featurize_datapoint(
                row[train_args.molecule_smiles_columns],
                train_args.number_of_molecules,
                row[train_args.target_column],
            )
            datapoints.extend([dp])
        except TypeError:  # dp returned None, a SMILES string could not be parsed
            bad_smiles_count += 1

    print(f"Dataset loaded. {bad_smiles_count} SMILES strings could not be parsed.")

    return Dataset(
        train_args,
        datapoints,
        get_features_to_normalize(train_args.atom_descriptors),
        train_args.number_of_molecules,
    )
