import numpy as np
from typing import Union
import torch


class SingleMolDatapoint:
    """
    Defines a single molecule datapoint. Contains a molecule's SMILES string, adjacency matrix, atom feature matrix,
    molecular features, and target to learn on.
    """

    def __init__(
        self,
        smiles: str,
        adjacency_matrix: torch.Tensor,
        atom_feature_matrix: torch.Tensor,
        molecule_features_vector: torch.Tensor,
        target: torch.float32 = None,
    ):
        self.smiles = smiles
        self.adjacency_matrix = adjacency_matrix
        self.atom_feature_matrix = atom_feature_matrix
        self.molecule_features_vector = molecule_features_vector
        self.target = target

    def to(self, device) -> None:
        self.adjacency_matrix.to(device)
        self.atom_feature_matrix.to(device)
        self.molecule_features_vector.to(device)
        # self.target.to(device)


class MultiMolDatapoint:
    """
    Defines a multi-molecule datapoint. Contains SMILES strings, adjacency matrices, atom feature matrices,
    and molecular features for each molecule in the datapoint, and contains the target to learn on.
    """

    def __init__(
        self,
        smiles_list: list[str],
        adjacency_matrices: list[torch.Tensor],
        atom_feature_matrices: list[torch.Tensor],
        molecule_feature_vectors: list[torch.Tensor],
        target: torch.float32 = None,
    ):
        self.smiles_list = smiles_list
        self.adjacency_matrices = adjacency_matrices
        self.atom_feature_matrices = atom_feature_matrices
        self.molecule_feature_vectors = molecule_feature_vectors
        self.target = target

    def __getitem__(self, idx):
        return (
            self.smiles_list[idx],
            self.adjacency_matrices[idx],
            self.atom_feature_matrices[idx],
            self.molecule_feature_vectors[idx],
        )

    def to(self, device) -> None:
        for i in range(len(self.smiles_list)):
            self.adjacency_matrices[i].to(device)
            self.atom_feature_matrices[i].to(device)
            self.molecule_feature_vectors[i].to(device)
        # self.target.to(device)
