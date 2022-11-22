from rdkit import Chem
from args import DataArgs
from data.featurizer import MoleculeFeaturizer
from data.dataset import load_dataset

if __name__ == "__main__":
    data_args = DataArgs().parse_args()
    dataset = load_dataset(data_args)
    featurizer = MoleculeFeaturizer(data_args)
    pass
