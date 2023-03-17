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

from collections import Counter
from scipy.ndimage import convolve1d
from utils.utils import get_lds_kernel_window

import numpy as np


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

        # For LDS, find the minimum target and the scaling factor such that target domain becomes [0, num_buckets)
        # 1. Set lower bound of targets to 1 by adding the minimum target to all targets if minimum target is less than 0
        targets = [dp.target for dp in self.datapoints]
        self.min_target = min(targets)
        if self.min_target < 0:
            targets = [t + self.min_target for t in targets]
        # Scale each target such that the maximum target = max bucket number = num_buckets
        max_target = max(targets)
        self.scale_factor = (train_args.lds_num_buckets - 1) / max_target

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

    def get_bin_idx(self, target: float) -> int:
        """
        Gets the bin index for a given target when all targets in the dataset are scaled to
        :param target: The target of a datapoint in the dataset.
        :return: The bin index of the target.
        """
        return int((target + -1 * self.min_target) * self.scale_factor)

    def create_lds_weights(self, kernel="gaussian", ks=5, sigma=2):
        """
        Creates a set of weights for each target in the dataset using LDS. See https://github.com/YyzHarry/imbalanced-regression
        :return:
        """
        bin_index_per_label = [self.get_bin_idx(dp.target) for dp in self.datapoints]
        Nb = max(bin_index_per_label) + 1
        num_samples_of_bins = dict(Counter(bin_index_per_label))
        emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]

        # lds_kernel_window: [ks,], here for example, we use gaussian, ks=5, sigma=2
        lds_kernel_window = get_lds_kernel_window(kernel, ks, sigma)
        # calculate effective label distribution: [Nb,]
        self.eff_label_dist = convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode="constant")


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
                row[train_args.target_column] if train_args.target_column else None,
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
