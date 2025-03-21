"""
Compute some molecular descriptors from RDkit molecules

- rdkit_to_array: helper function to convert RDkit fingerprints to a numpy array
- mols_to_maccs: Get MACCs key descriptors from a list of RDKit molecule objects
- mols_to_ecfp: Get ECFPs from a list of RDKit molecule objects
- mols_to_descriptors: Get the full set of available RDKit descriptors (normalized) for a list of RDKit molecule objects


Derek van Tilborg
Eindhoven University of Technology
June 2024
"""

from typing import Union
from warnings import warn
import numpy as np
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit.Chem import MACCSkeys, Descriptors
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.rdMolDescriptors import CalcNumRings
from constants import VOCAB
from cheminformatics.encoding import smiles_tokenizer
from cheminformatics.cats import cats


def rdkit_to_array(fp: list) -> np.ndarray:
    """ Convert a list of RDkit fingerprint objects into a numpy array """
    output = []
    for f in fp:
        arr = np.zeros((1,))
        ConvertToNumpyArray(f, arr)
        output.append(arr)
    return np.asarray(output)


def mols_to_maccs(mols: list[Mol], progressbar: bool = False, to_array: bool = False) -> Union[list, np.ndarray]:
    """ Get MACCs key descriptors from a list of RDKit molecule objects

    :param mols: list of RDKit mol objects, e.g., as obtained through smiles_to_mols()
    :param progressbar: toggles progressbar (default = False)
    :param to_array: Toggles conversion of RDKit fingerprint objects to a Numpy Array (default = False)
    :return: Numpy Array of MACCs keys
    """
    was_list = True if type(mols) is list else False
    mols = mols if was_list else [mols]
    fp = [MACCSkeys.GenMACCSKeys(m) for m in tqdm(mols, disable=not progressbar)]
    if not to_array:
        return fp if was_list else fp[0]
    return rdkit_to_array(fp)


def mols_to_ecfp(mols: list[Mol], radius: int = 2, nbits: int = 2048, progressbar: bool = False,
                 to_array: bool = False) -> Union[list, np.ndarray]:
    """ Get ECFPs from a list of RDKit molecule objects

    :param mols: list of RDKit mol objects, e.g., as obtained through smiles_to_mols()
    :param radius: Radius of the ECFP (default = 2)
    :param nbits: Number of bits (default = 2048)
    :param progressbar: toggles progressbar (default = False)
    :param to_array: Toggles conversion of RDKit fingerprint objects to a Numpy Array (default = False)
    :return: list of RDKit ECFP fingerprint objects, or a Numpy Array of ECFPs if to_array=True
    """
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)

    was_list = True if type(mols) is list else False
    mols = mols if was_list else [mols]
    fp = [mfpgen.GetFingerprint(m) for m in tqdm(mols, disable=not progressbar)]
    if not to_array:
        return fp if was_list else fp[0]
    return rdkit_to_array(fp)


def mols_to_cats(mols: list[Mol], progressbar: bool = False) -> Union[list, np.ndarray]:
    """ Get CATs pharmacophore descriptors from a list of RDKit molecule objects (implementation from Alex Müller)

    Descriptions of the individual features can be obtained with ``cheminformatics.cats.get_cats_sigfactory``.

    :param mols: list of RDKit mol objects, e.g., as obtained through smiles_to_mols()
    :param progressbar: toggles progressbar (default = False)
    :return: a Numpy Array of CATs
    """

    was_list = True if type(mols) is list else False
    mols = mols if was_list else [mols]
    cats_list = [cats(m) for m in tqdm(mols, disable=not progressbar)]

    return np.array(cats_list)


def mols_to_descriptors(mols: list[Mol], progressbar: bool = False, normalize: bool = True) -> np.ndarray:
    """ Get the full set of available RDKit descriptors for a list of RDKit molecule objects

    :param mols: list of RDKit mol objects, e.g., as obtained through smiles_to_mols()
    :param progressbar: toggles progressbar (default = False)
    :param normalize: toggles min-max normalization
    :return: Numpy Array of all RDKit descriptors
    """
    mols = [mols] if type(mols) is not list else mols
    x = np.array([list(Descriptors.CalcMolDescriptors(m).values()) for m in tqdm(mols, disable=not progressbar)])
    if normalize:
        x = max_normalization(x)
        if np.isnan(x).any():
            warn("There were some nan-values introduced by 0-columns. Replaced all nan-values with 0")
            x = np.nan_to_num(x, nan=0)

    return x


def max_normalization(x: np.ndarray) -> np.ndarray:
    """ Perform max normalization on a matrix x / x.max(axis=0), just like
    sklearn.preprocessing.normalize(x, axis=0, norm='max')

    :param x: array to be normalized
    :return: normalized array
    """
    return x / x.max(axis=0)



def n_smiles_tokens_no_specials(smi: str) -> int:
    """Converts a SMILES string into a list of token indices using a predefined vocabulary. No special tokens are used
     """

    return len(smiles_tokenizer(smi))


def n_smiles_branches(smi: str) -> int:
    """ Counts the number of branches in a SMILES string """

    return smi.count('(')


def mol_weight(smi: str) -> float:
    """ Gets the molecular weight of a molecule from its SMILES """

    return ExactMolWt(Chem.MolFromSmiles(smi))


def num_rings(smi: str) -> int:
    """ Counts the number of rings """

    return CalcNumRings(Chem.MolFromSmiles(smi))
