import functools

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Lipinski, rdMolDescriptors, rdPartialCharges
from rdkit.Chem.rdchem import HybridizationType

"""
Much of the code below is adapted from scikit-chem.

See https://scikit-chem.readthedocs.io/en/stable/_modules/skchem/descriptors/atom.html for source.
"""

MAX_ATOMIC_NUM = 118

# Maps certain atomic features values to their idx for one-hot encoding
# Adapted from https://github.com/chemprop/chemprop/blob/98fc25eec8c57171c56d6ffad03ae1336c016e3b/chemprop/features/featurization.py#L174
CATEGORICAL_FEATURE_VAL_TO_IDX = {
    "atomic_number": [""] * MAX_ATOMIC_NUM,
    "degree": [0, 1, 2, 3, 4, 5],
    "formal_charge": [-1, -2, 1, 2, 0],
    "num_Hs": [0, 1, 2, 3, 4],
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
}


def get_descriptor_vector_length(feature: str) -> int:
    """
    Calculates the length of a feature vector for a single feature.
    :param feature: The name of the descriptor to calculate the feature vector length for.
    :return: The length of the feature vector length.
    """
    if feature not in CATEGORICAL_FEATURE_VAL_TO_IDX:
        return 1
    else:
        return len(CATEGORICAL_FEATURE_VAL_TO_IDX[feature])


def get_features_vector_length(descriptors: list[str]) -> int:
    """
    Calculates the length of the features vector given a set number of descriptors.
        :param descriptors: The list of descriptors for which the length of the features vector will be calculated. Features
                         that are not in CATEGORICAL_FEATURE_VAL_TO_IDX are assumed to have length 1
        :return: The length of the features vector as an integeger.
    """
    length = 0
    for descriptor in descriptors:
        length += get_descriptor_vector_length(descriptor)
    return length


def get_features_to_normalize(dscriptors: list[str]) -> list[bool]:
    """
    Returns a boolean vector that denotes which features in a feature vector generated from a set of descriptors can or
    should be normalized.
    :param dscriptors: A list of descriptors.
    :return: A boolean vector that denotes which features in a feature vector generated from a set of descriptors can or
    should be normalized.
    """
    normalize_bool_vector = []
    for descriptor in dscriptors:
        normalize_bool_vector.extend([descriptor in CONTINUOUS_DESCRIPTORS] * get_descriptor_vector_length(descriptor))
    return normalize_bool_vector


def one_hot_encode(idx: int, max_values: int) -> list[bool]:
    """
    Constructs a list of length max_values, setting idx to 1

    :param idx: The index of the feature to be "hot"
    :param max_values: The maximum number of possible values the feature can be
    :return: A list of 0s of length max_values, with the value at index idx set to 1.
    """
    l = [False] * max_values
    l[idx] = True
    return l


def atomic_number(a: Chem.rdchem.Atom) -> list[int]:
    """Atomic number of atom"""

    atomic_num = a.GetAtomicNum()
    feature_vec = one_hot_encode(atomic_num, MAX_ATOMIC_NUM)
    return feature_vec


def is_h_acceptor(a: Chem.rdchem.Atom) -> list[bool]:

    """Is an H acceptor?"""

    m = a.GetOwningMol()
    idx = a.GetIdx()
    return [idx in [i[0] for i in Lipinski._HAcceptors(m)]]


def is_h_donor(a: Chem.rdchem.Atom) -> list[bool]:

    """Is an H donor?"""

    m = a.GetOwningMol()
    idx = a.GetIdx()
    return [idx in [i[0] for i in Lipinski._HDonors(m)]]


def is_hetero(a: Chem.rdchem.Atom) -> list[bool]:

    """Is a heteroatom?"""

    m = a.GetOwningMol()
    idx = a.GetIdx()
    return [idx in [i[0] for i in Lipinski._Heteroatoms(m)]]


def atomic_mass(a: Chem.rdchem.Atom) -> list[float]:

    """Atomic mass of atom"""

    return [a.GetMass()]


def valence(a: Chem.rdchem.Atom) -> list[int]:

    """returns the valence of the atom"""
    explicit_valence = a.GetExplicitValence()
    implicit_valence = a.GetImplicitValence()
    return [explicit_valence(a)[0] + implicit_valence(a)[0]]


def formal_charge(a: Chem.rdchem.Atom) -> list[bool]:

    """Formal charge of atom"""

    charge = a.GetFormalCharge()

    charge_idx = CATEGORICAL_FEATURE_VAL_TO_IDX["formal_charge"].index(charge)
    possible_formal_charge_values = len(CATEGORICAL_FEATURE_VAL_TO_IDX["formal_charge"])
    formal_charge_vec = one_hot_encode(charge_idx, possible_formal_charge_values)

    return formal_charge_vec


def is_aromatic(a: Chem.rdchem.Atom) -> list[bool]:

    """Boolean if atom is aromatic"""

    return [a.GetIsAromatic()]


def num_hydrogens(a: Chem.rdchem.Atom) -> list[int]:

    """Number of hydrogens"""
    num_implicit_hydrogens = a.GetNumImplicitHs()
    num_explicit_hydrogens = a.GetNumExplicitHs()
    num_hydrogens = num_explicit_hydrogens + num_implicit_hydrogens

    num_hydrogens_idx = CATEGORICAL_FEATURE_VAL_TO_IDX["num_Hs"].index(num_hydrogens)
    possible_num_hydrogens_values = len(CATEGORICAL_FEATURE_VAL_TO_IDX["num_Hs"])
    num_hydrogens_vec = one_hot_encode(num_hydrogens_idx, possible_num_hydrogens_values)

    return num_hydrogens_vec


def is_in_ring(a: Chem.rdchem.Atom) -> list[int]:

    """Whether the atom is in a ring"""

    return [a.IsInRing()]


def crippen_log_p_contrib(a: Chem.rdchem.Atom) -> list[float]:

    """Hacky way of getting logP contribution."""

    idx = a.GetIdx()
    m = a.GetOwningMol()
    return [Crippen._GetAtomContribs(m)[idx][0]]


def crippen_molar_refractivity_contrib(a: Chem.rdchem.Atom) -> list[float]:

    """Hacky way of getting molar refractivity contribution."""

    idx = a.GetIdx()
    m = a.GetOwningMol()
    return [Crippen._GetAtomContribs(m)[idx][1]]


def tpsa_contrib(a: Chem.rdchem.Atom) -> list[float]:

    """Hacky way of getting total polar surface area contribution."""

    idx = a.GetIdx()
    m = a.GetOwningMol()
    return [rdMolDescriptors._CalcTPSAContribs(m)[idx]]


def labute_asa_contrib(a: Chem.rdchem.Atom) -> list[float]:

    """Hacky way of getting accessible surface area contribution."""

    idx = a.GetIdx()
    m = a.GetOwningMol()
    return [rdMolDescriptors._CalcLabuteASAContribs(m)[0][idx]]


# TODO: Fix, doesn't work right now
# def gasteiger_charge(a, force_calc=False) -> list[float]:
#
#     """Hacky way of getting gasteiger charge"""
#
#     res = a.GetPropsAsDict()["_GasteigerCharge"]
#     if res and not force_calc:
#         return [float(res)]
#     else:
#         m = a.GetOwningMol()
#         rdPartialCharges.ComputeGasteigerCharges(m)
#         return [float(a.props["_GasteigerCharge"])]


def hybridization(a) -> list[bool]:

    """Hybridized as type hybrid_type, default SP3"""

    return [
        a.GetHybridization() is hybridization_type
        for hybridization_type in CATEGORICAL_FEATURE_VAL_TO_IDX["hybridization"]
    ]


# Contains discrete or categorical descriptors that should not be normalized.
DISCRETE_DESCRIPTORS = {
    "atomic_number": atomic_number,
    "formal_charge": formal_charge,
    "valence": valence,
    "is_aromatic": is_aromatic,
    "num_hydrogens": num_hydrogens,
    "is_in_ring": is_in_ring,
    "is_h_acceptor": is_h_acceptor,
    "is_h_donor": is_h_donor,
    "is_heteroatom": is_hetero,
}

# Contains continuous descriptors that should be normalized.
CONTINUOUS_DESCRIPTORS = {
    "atomic_mass": atomic_mass,
    "log_p_contrib": crippen_log_p_contrib,
    "molar_refractivity_contrib": crippen_molar_refractivity_contrib,
    "total_polar_surface_area_contrib": tpsa_contrib,
    "total_labute_accessible_surface_area": labute_asa_contrib,
}


def all_descriptors():
    return DISCRETE_DESCRIPTORS | CONTINUOUS_DESCRIPTORS
