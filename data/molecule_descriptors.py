from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import numpy as np

FINGERPRINT_SIZE = 2048

rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=FINGERPRINT_SIZE)
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(fpSize=FINGERPRINT_SIZE)


def rdkit_fingerprint(mol: Chem.Mol) -> np.array:
    """
    Generates an rdkit/Daylight-like fingerprint for a given molecule.
    :param mol: An rdkit molecule.
    :return: An np.array containing the fingerprint bit vector.
    """
    return rdkit_gen.GetFingerprintAsNumPy(mol)


def morgan_fingerprint(mol: Chem.Mol) -> np.array:
    """
    Generates a morgan similarity fingerprint for a given molecule.
    :param mol: An rdkit molecule.
    :return: An np.array containing the fingerprint bit vector.
    """
    return morgan_gen.GetFingerprintAsNumPy(mol)


# Contains discrete or categorical descriptors that should not be normalized.
DISCRETE_DESCRIPTORS = {
    "morgan": morgan_fingerprint,
    "rdkit": rdkit_fingerprint,
}


def all_descriptors():
    return DISCRETE_DESCRIPTORS
