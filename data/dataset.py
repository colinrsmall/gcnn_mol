import numpy as np

from data.moldata import SingleMolDatapoint, MultiMolDatapoint, AbstractDatapoint
from args import TrainArgs
from typing import Union
from data.featurizer import MoleculeFeaturizer
import pandas as pd
from data.atom_descriptors import get_features_vector_length, get_features_to_normalize
from tqdm.auto import tqdm
from torch.utils import data
import torch
from sklearn.preprocessing import StandardScaler


class Dataset(data.Dataset):
    # TODO: Inherit from nn.Dataset (https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
    def __init__(
        self,
        train_args: TrainArgs,
        datapoints: list[Union[SingleMolDatapoint, MultiMolDatapoint]],
        atom_feature_vector_length: int,
        mol_feature_vector_length: int,
        features_to_normalize: list[bool],
        num_molecules_per_datapoint: int,
    ):
        self.data_args = train_args
        self.datapoints = datapoints
        self.atom_feature_vector_length = atom_feature_vector_length
        self.mol_feature_vector_length = mol_feature_vector_length
        self.features_to_normalize = features_to_normalize
        self.num_molecules_per_datapoint = num_molecules_per_datapoint

    def fit_scalers_to_features(self) -> list[StandardScaler]:
        """
        Fit a set of scalers to the features of the dataset.
        :return: A list that matches the length of the feature vectors generated for each atom. Contains None at indices
        corresponding to features that should not be scaled or an sklearn StandardScaler for features that should be.
        """

        # Initialize scaler and stack each molecule datapoints' atom feature matrices into a single matrix
        scalers = [(StandardScaler() if f else None) for f in self.features_to_normalize]
        atom_features_stack = torch.vstack([torch.vstack(dp.atom_feature_matrices) for dp in self.datapoints])

        # Fit each scaler
        for i, feature in enumerate(self.features_to_normalize):
            if feature:
                feature_vector = atom_features_stack[:, i]
                scalers[i].fit(feature_vector.reshape([-1, 1]))

        return scalers

    # def fit_scalers_to_target(self) -> StandardScaler:
    #     """
    #     Fits a single scaler to the targets of the dataset.
    #     :return: An sklearn StandardScaler fit to the targets of the dataset.
    #     """
    #     scaler = StandardScaler()
    #     mol_target_stack = np.array([dp.target for dp in self.datapoints])
    #     scaler.fit(mol_target_stack.reshape(-1, 1))
    #     return scaler

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

    # def normalize_targets(self, scaler: StandardScaler) -> None:
    #     """
    #     Normalize the targets of the dataset in place using a provided scaler.
    #     :param scaler: An sklearn StandardScaler previously fit to the targets of a dataset.
    #     """
    #     for dp in self.datapoints:
    #         dp.target = scaler.transform(dp.target)

    def to(self, device: torch.device) -> None:
        """
        Send the tensors and targets contained in the dataset to a provided device.
        :param device: A torch.device to send the tensors to.
        """
        for dp in self.datapoints:
            dp.to(device)

    def __len__(self) -> int:
        return len(self.datapoints)

    def __getitem__(self, idx) -> AbstractDatapoint:
        return self.datapoints[idx]


def load_dataset(train_args: TrainArgs) -> Dataset:
    featurizer = MoleculeFeaturizer(train_args)
    data = pd.read_csv(train_args.dataset_path)

    datapoints = []
    print("Loading and featurizing molecules:")
    for _, row in tqdm(data.iterrows(), total=len(list(data.iterrows()))):
        datapoints.extend(
            [
                featurizer.featurize_datapoint(
                    row[train_args.molecule_smiles_columns],
                    train_args.number_of_molecules,
                    row[train_args.target_column],
                )
            ]
        )

    return Dataset(
        train_args,
        datapoints,
        get_features_vector_length(train_args.atom_descriptors),
        1,
        get_features_to_normalize(train_args.atom_descriptors),
        train_args.number_of_molecules,
    )
