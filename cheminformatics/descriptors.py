"""
Compute some molecular descriptors from RDkit molecules

- rdkit_to_array: helper function to convert RDkit fingerprints to a numpy array
- mols_to_maccs: Get MACCs key descriptors from a list of RDKit molecule objects
- mols_to_ecfp: Get ECFPs from a list of RDKit molecule objects
- mols_to_descriptors: Get the full set of available RDKit descriptors (normalized) for a list of RDKit molecule objects


Derek van Tilborg
Eindhoven University of Technology
Jan 2024
"""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from typing import Union
from rdkit.Chem.rdchem import Mol
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem import Descriptors
from warnings import warn
from constants import VOCAB
from cheminformatics.utils import smiles_tokenizer


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
    was_list = True if type(mols) is list else False
    mols = mols if was_list else [mols]
    fp = [GetMorganFingerprintAsBitVect(m, radius, nBits=nbits) for m in tqdm(mols, disable=not progressbar)]
    if not to_array:
        return fp if was_list else fp[0]
    return rdkit_to_array(fp)


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


def smiles_to_encoding(smi: str) -> torch.Tensor:
    """Converts a SMILES string into a list of token indices using a predefined vocabulary """

    encoding = [VOCAB['start_idx']] + [VOCAB['token_indices'][i] for i in smiles_tokenizer(smi)] + [VOCAB['end_idx']]
    encoding.extend([VOCAB['pad_idx']] * (VOCAB['max_len'] - len(encoding)))

    return torch.tensor(encoding)


def encode_smiles(smiles: list[str]):
    return torch.stack([smiles_to_encoding(smi) for smi in smiles])


def one_hot_encode(encodings):
    return F.one_hot(encodings, VOCAB['vocab_size'])


def probs_to_encoding(x: torch.Tensor) -> torch.Tensor:
    """ Gets the most probable token for every entry in a sequence

    :param x: Tensor in shape (batch x seq_length x vocab)
    :return: x: Tensor in shape (batch x seq_length)
    """

    assert x.dim() == 3
    return x.argmax(dim=2)


def encoding_to_smiles(encoding: torch.Tensor) -> list[str]:
    """ Convert a tensor of token indices into a list of character strings

    :param encoding: Tensor in shape (batch x seq_length x vocab) containing ints
    :return: list of SMILES strings (with utility tokens)
    """

    assert encoding.dim() == 2, f"Encodings should be shape (batch_size x seq_length), not {encoding.shape}"
    return [''.join([VOCAB['indices_token'][t_i.item()] for t_i in enc]) for enc in encoding]


def clean_smiles(smiles: list[str]) -> list[str]:
    """ Strips the start and end character from a list of SMILES strings

    :param smiles: list of 'uncleaned' SMILES
    :return: list of SMILES strings
    """

    return [smi.split(VOCAB['start_char'])[-1].split(VOCAB['end_char'])[0] for smi in smiles]
