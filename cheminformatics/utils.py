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
from collections import defaultdict
import numpy as np
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.Scaffolds import MurckoScaffold
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


def get_scaffold(mol, scaffold_type: str = 'bemis_murcko'):
    """ Get the molecular scaffold from a molecule. Supports four different scaffold types:
            `bemis_murcko`: RDKit implementation of the bemis-murcko scaffold; a scaffold of rings and linkers, retains
            some sidechains and ring-bonded substituents.
            `bemis_murcko_bajorath`: Rings and linkers only, with no sidechains.
            `generic`: Bemis-Murcko scaffold where all atoms are carbons & bonds are single, i.e., a molecular skeleton.
            `cyclic_skeleton`: A molecular skeleton w/o any sidechains, only preserves ring structures and linkers.

    Examples:
        original molecule: 'CCCN(Cc1ccccn1)C(=O)c1cc(C)cc(OCCCON=C(N)N)c1'
        Bemis-Murcko scaffold: 'O=C(NCc1ccccn1)c1ccccc1'
        Bemis-Murcko-Bajorath scaffold:' c1ccc(CNCc2ccccn2)cc1'
        Generic RDKit: 'CC(CCC1CCCCC1)C1CCCCC1'
        Cyclic skeleton: 'C1CCC(CCCC2CCCCC2)CC1'

    :param mol: RDKit mol object
    :param scaffold_type: 'bemis_murcko' (default), 'bemis_murcko_bajorath', 'generic', 'cyclic_skeleton'
    :return: RDKit mol object
    """
    all_scaffs = ['bemis_murcko', 'bemis_murcko_bajorath', 'generic', 'cyclic_skeleton']
    assert scaffold_type in all_scaffs, f"scaffold_type='{scaffold_type}' is not supported. Pick from: {all_scaffs}"

    # designed to match atoms that are doubly bonded to another atom.
    PATT = Chem.MolFromSmarts("[$([D1]=[*])]")
    # replacement SMARTS (matches any atom)
    REPL = Chem.MolFromSmarts("[*]")

    Chem.RemoveStereochemistry(mol)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)

    if scaffold_type == 'bemis_murcko':
        return scaffold

    if scaffold_type == 'bemis_murcko_bajorath':
        scaffold = AllChem.DeleteSubstructs(scaffold, PATT)
        return scaffold

    if scaffold_type == 'generic':
        scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        return scaffold

    if scaffold_type == 'cyclic_skeleton':
        scaffold = AllChem.ReplaceSubstructs(scaffold, PATT, REPL, replaceAll=True)[0]
        scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        scaffold = MurckoScaffold.GetScaffoldForMol(scaffold)
        return scaffold


def mols_to_scaffolds(mols: list[Mol], scaffold_type: str = 'bemis_murcko') -> list:
    """ Convert a list of RDKit molecules objects into scaffolds. See cheminformatics.utils.get_scaffold

    :param mols: list of RDKit mol objects, e.g., as obtained through smiles_to_mols()
    :param scaffold_type: 'bemis_murcko' (default), 'bemis_murcko_bajorath', 'generic', 'cyclic_skeleton'
    :return: RDKit mol objects of the scaffolds
    """
    scaffolds = [get_scaffold(m, scaffold_type) for m in mols]

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


def smile_complexity(smile: str) -> float:
    """ Calculates Bertz CT as a measure of molecular complexity

    RDKit: Bertz CT consists of a sum of two terms, one representing the complexity of the bonding, the other
    representing the complexity of the distribution of heteroatoms.

    From S. H. Bertz, J. Am. Chem. Soc., vol 103, 3599-3601 (1981)

    :param smile: SMILES string, e.g.; 'CC(C)C1CN=C(N)C1'
    :return: estimated complexity

    """
    mol = Chem.MolFromSmiles(smile)
    return Chem.GraphDescriptors.BertzCT(mol)
