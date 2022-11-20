from rdkit import Chem
from args import DataArgs
from data.featurizer import MoleculeFeaturizer

if __name__ == "__main__":
    arg = DataArgs().parse_args()
    mol = Chem.MolFromSmiles("O=C(O)c1ccccc1")
    featurizer = MoleculeFeaturizer(arg)
    moldata = featurizer.featurize_mol(mol)
    pass
