"""
A collection of utility functions

- canonicalize_smiles: Canonicalize a list of SMILES strings with the RDKit SMILES canonicalization algorithm
- smiles_to_mols: Convert a list of SMILES strings to RDkit molecules (and sanitize them)
- mols_to_smiles: Convert a list of RDkit molecules back into SMILES strings
- mols_to_scaffolds: Convert a list of RDKit molecules objects into scaffolds (bismurcko or bismurcko_generic)
- map_scaffolds: Find which molecules share the same scaffold
- smiles_tokenizer: tokenize a SMILES strings into individual characters

Derek van Tilborg
Eindhoven University of Technology
Jan 2024
"""

import re
from typing import Union
from collections import defaultdict
import numpy as np
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol, MakeScaffoldGeneric
from rdkit.DataStructs import BulkTanimotoSimilarity
import torch
import torch.nn.functional as F
from constants import VOCAB


def canonicalize_smiles(smiles: Union[str, list[str]]) -> Union[str, list[str]]:
    """ Canonicalize a list of SMILES strings with the RDKit SMILES canonicalization algorithm """
    if type(smiles) is str:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

    return [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in smiles]


def smiles_to_mols(smiles: list[str], sanitize: bool = True, partial_charges: bool = False) -> list:
    """ Convert a list of SMILES strings to RDkit molecules (and sanitize them)

    :param smiles: List of SMILES strings
    :param sanitize: toggles sanitization of the molecule. Defaults to True.
    :param partial_charges: toggles the computation of partial charges (default = False)
    :return: List of RDKit mol objects
    """
    mols = []
    was_list = True
    if type(smiles) is str:
        was_list = False
        smiles = [smiles]

    for smi in smiles:
        molecule = Chem.MolFromSmiles(smi, sanitize=sanitize)

        # If sanitization is unsuccessful, catch the error, and try again without
        # the sanitization step that caused the error
        if sanitize:
            flag = Chem.SanitizeMol(molecule, catchErrors=True)
            if flag != Chem.SanitizeFlags.SANITIZE_NONE:
                Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag)

        Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)

        if partial_charges:
            Chem.rdPartialCharges.ComputeGasteigerCharges(molecule)

        mols.append(molecule)

    return mols if was_list else mols[0]


def mols_to_smiles(mols: list[Mol]) -> list[str]:
    """ Convert a list of RDKit molecules objects into a list of SMILES strings"""
    return [Chem.MolToSmiles(m) for m in mols] if type(mols) is list else Chem.MolToSmiles(mols)


def mols_to_scaffolds(mols: list[Mol], scaffold_type: str = 'bismurcko') -> list:
    """ Convert a list of RDKit molecules objects into scaffolds (bismurcko or bismurcko_generic)

    :param mols: list of RDKit mol objects, e.g., as obtained through smiles_to_mols()
    :param scaffold_type: type of scaffold: bismurcko, bismurcko_generic (default = 'bismurcko')
    :return: RDKit mol objects of the scaffolds
    """
    if scaffold_type == 'bismurcko_generic':
        scaffolds = [MakeScaffoldGeneric(m) for m in mols]
    else:
        scaffolds = [GetScaffoldForMol(m) for m in mols]

    return scaffolds


def tanimoto_matrix(fingerprints: list, progressbar: bool = False, fill_diagonal: bool = True, dtype=np.float16) \
        -> np.ndarray:
    """

    :param fingerprints: list of RDKit fingerprints
    :param progressbar: toggles progressbar (default = False)
    :param dtype: numpy dtype (default = np.float16)
    :param fill_diagonal: Fill the diagonal with 1's (default = True)

    :return: Tanimoto similarity matrix
    """
    n = len(fingerprints)

    X = np.zeros([n, n], dtype=dtype)
    # Fill the upper triangle of the pairwise matrix
    for i in tqdm(range(n), disable=not progressbar, desc=f"Computing pairwise Tanimoto similarity of {n} molecules"):
        X[i, i+1:] = BulkTanimotoSimilarity(fingerprints[i], fingerprints[i+1:])
    # Mirror out the lower triangle
    X = X + X.T - np.diag(np.diag(X))

    if fill_diagonal:
        np.fill_diagonal(X, 1)

    return X


def smiles_tokenizer(smiles: str, extra_patterns: list[str] = None) -> list[str]:
    """ Tokenize a SMILES. By default, we use the base SMILES grammar tokens and the reactive nonmetals H, C, N, O, F,
    P, S, Cl, Se, Br, I:

    '(\\[|\\]|Cl|Se|se|Br|H|C|c|N|n|O|o|F|P|p|S|s|I|\\(|\\)|\\.|=|#|-|\\+|\\\\|\\/|:|~|@|\\?|>|\\*|\\$|\\%\\d{2}|\\d)'

    :param smiles: SMILES string
    :param extra_patterns: extra tokens to consider (default = None)
        e.g. metalloids: ['Si', 'As', 'Te', 'te', 'B', 'b']  (in ChEMBL33: B+b=0.23%, Si=0.13%, As=0.01%, Te+te=0.01%).
        Mind you that the order matters. If you place 'C' before 'Cl', all Cl tokens will actually be tokenized as C,
        meaning that subsets should always come after superset strings, aka, place two letter elements first in the list
    :return: list of tokens extracted from the smiles string in their original order
    """
    base_smiles_patterns = "(\[|\]|insert_here|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\d)"
    # reactive_nonmetals = ['Cl', 'Si', 'si', 'Se', 'se', 'Br', 'B', 'H', 'C', 'c', 'N', 'n', 'O', 'o', 'F', 'P', 'p', 'S', 's', 'I']
    reactive_nonmetals = ['Cl', 'Br', 'H', 'C', 'c', 'N', 'n', 'O', 'o', 'F', 'P', 'p', 'S', 's', 'I']

    # Add all allowed elements to the base SMILES tokens
    extra_patterns = reactive_nonmetals if extra_patterns is None else extra_patterns + reactive_nonmetals
    pattern = base_smiles_patterns.replace('insert_here', "|".join(extra_patterns))

    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smiles)]

    return tokens


def map_scaffolds(smiles: list[str]) -> (list, dict[str, list[int]]):
    """ Find which molecules share the same scaffold

    :param mols: RDKit mol objects, e.g., as obtained through smiles_to_mols()
    :return: scaffolds, dict of unique scaffolds and which molecules (indices) share them -> {'c1ccccc1': [0, 12, 47]}
    """

    scaffolds = []
    for smi in smiles:
        scaff_smi = mols_to_smiles(mols_to_scaffolds([smiles_to_mols(smi)])[0])
        scaffolds.append(scaff_smi)

    uniques = defaultdict(list)
    for i, s in enumerate(scaffolds):
        uniques[s].append(i)

    return scaffolds, uniques



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
