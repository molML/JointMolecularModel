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

from typing import Union
import numpy as np
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol, MakeScaffoldGeneric
from rdkit.DataStructs import BulkTanimotoSimilarity


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

    return mols


def mols_to_smiles(mols: list[Mol]) -> list[str]:
    """ Convert a list of RDKit molecules objects into a list of SMILES strings"""
    return [Chem.MolToSmiles(m) for m in mols]


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
