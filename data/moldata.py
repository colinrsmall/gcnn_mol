import numpy as np
from typing import Union


class SingleMolDatapoint:
    """
    Defines a single molecule datapoint. Contains a molecule's SMILES string, adjacency matrix, atom feature matrix,
    molecular features, and target to learn on.
    """

    def __init__(
        self,
        smiles: str,
        adjacency_matrix: np.ndarray,
        atom_feature_matrix: np.ndarray,
        molecule_features: np.ndarray,
        target: Union[int, float] = None,
    ):
        self.smiles = smiles
        self.adjacency_matrix = adjacency_matrix
        self.atom_feature_matrix = atom_feature_matrix
        self.molecule_features = molecule_features
        self.target = target


class MultiMolDatapoint:
    """
    Defines a multi-molecule datapoint. Contains SMILES strings, adjacency matrices, atom feature matrices,
    and molecular features for each molecule in the datapoint, and contains the target to learn on.
    """

    def __init__(
        self,
        smiles_list: list[str],
        adjacency_matrices: list[np.ndarray],
        atom_feature_matrices: list[np.ndarray],
        molecule_feature_vectors: list[np.ndarray],
        target: Union[int, float] = None,
    ):
        self.smiles_list = smiles_list
        self.adjacency_matrices = adjacency_matrices
        self.atom_feature_matrices = atom_feature_matrices
        self.molecule_feature_vectors = molecule_feature_vectors
        self.target = target
