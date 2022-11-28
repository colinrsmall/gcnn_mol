import numpy as np
import numpy.typing as npt
import torch
from rdkit.Chem.rdchem import Atom, Mol
from rdkit.Chem import GetAdjacencyMatrix
from rdkit.Chem import MolToSmiles, MolFromSmiles

from data import atom_descriptors
from args import TrainArgs
from data.moldata import SingleMolDatapoint, MultiMolDatapoint

from typing import Union, Optional, Tuple


class MoleculeFeaturizer:
    def __init__(self, args: TrainArgs):
        self.data_args = args

    def create_descriptors_for_atom(self, atom: Atom) -> (np.ndarray, np.ndarray):
        """
        Creates descriptor feature vector for a given atom using the descriptors provided in the featurizer's
        featurization args.
        :param atom: An atom to create descriptor features for.
        :return: An np.ndarray containing the descriptor features for the atom
        """
        feature_vector = []
        for descriptor in self.data_args.atom_descriptors:
            features = atom_descriptors.all_descriptors()[descriptor](atom)
            feature_vector.extend(features)
        return np.array(feature_vector)

    def _featurize_mol(self, smiles: str) -> Optional[Tuple[str, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]]:
        """
        Featurize an individual molecule. Generates both an adjacency matrix representing the bonds of the molecule and
        a features matrix containing the calculated descriptors for each atom in the molecule. Saves both in a MolData
        object.
        :param smiles: The SMILES string of a molecule to featurize.
        :return: A MolData object containing the adjacency matrix and features matrix.
        """
        try:
            mol = MolFromSmiles(smiles)
        except TypeError as e:  # Molecule's SMILES string could not be parsed
            raise e

        # Return None if the molecule's SMILES string could not be parsed
        if mol is None:
            return None

        # Generate adjacency matrix
        adjacency_matrix = GetAdjacencyMatrix(mol)

        # Check for improper adjacency matrices
        if 0 in np.sum(adjacency_matrix, axis=1):
            print(
                f"Adjacency matrix for {smiles} contains a 0, suggesting the SMILES string could not be parsed correctly."
            )
            return None

        # Add self-connections to the adjacency matrix
        np.fill_diagonal(adjacency_matrix, 1)

        # Turn adjacency matrix into a torch.Tensor
        adjacency_matrix = torch.from_numpy(adjacency_matrix).to(torch.float32)

        # Generate atom features
        # features_vector_length = atom_descriptors.get_features_vector_length(self.data_args.atom_descriptors)
        # Compute feature vector length by featurizing dummy atom
        features_vector_length = len(self.create_descriptors_for_atom(mol.GetAtoms()[0]))

        atom_feature_matrix = np.zeros((len(adjacency_matrix), features_vector_length))
        atom_feature_matrix = torch.from_numpy(atom_feature_matrix).to(torch.float32)

        for atom in mol.GetAtoms():
            atom_feature_matrix[atom.GetIdx()] = torch.from_numpy(self.create_descriptors_for_atom(atom))

        # Generate molecule features
        # TODO: Implement, call from data.molecule_descriptors
        mol_features = torch.from_numpy(np.zeros(1)).to(torch.float32)

        return smiles, adjacency_matrix, atom_feature_matrix, mol_features

    def featurize_datapoint(self, smiles: Union[str, list[str]], number_of_molecules: int, target: float):

        if number_of_molecules == 1:
            smiles, adjacency_matrix, atom_feature_matrix, mol_features = self._featurize_mol(smiles)
            return SingleMolDatapoint(smiles, adjacency_matrix, atom_feature_matrix, mol_features, target)

        else:
            smiles_list = []
            adjacency_matrices = []
            atom_feature_matrices = []
            mol_feature_vectors = []

            for smiles in smiles:
                smiles, adjacency_matrix, atom_feature_matrix, mol_features = self._featurize_mol(smiles)
                smiles_list.append(smiles)
                adjacency_matrices.append(adjacency_matrix)
                atom_feature_matrices.append(atom_feature_matrix)
                mol_feature_vectors.append(mol_features)

            return MultiMolDatapoint(
                smiles_list, adjacency_matrices, atom_feature_matrices, mol_feature_vectors, target
            )
