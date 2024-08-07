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
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.Scaffolds import MurckoScaffold
from cheminformatics.encoding import smiles_tokenizer
from constants import VOCAB


def canonicalize_smiles(smiles: Union[str, list[str]]) -> Union[str, list[str]]:
    """ Canonicalize a list of SMILES strings with the RDKit SMILES canonicalization algorithm """
    if type(smiles) is str:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

    return [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in smiles]


def randomize_smiles_string(smi: str) -> str:
    """ Randomize a SMILES string. Check if the new SMILES is not out of bounds with our vocab rules

    :param smi: SMILES string
    :return: randomized SMILES string
    """
    random_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=False, doRandom=True)

    if smiles_fits_in_vocab(random_smi):
        return random_smi

    return smi


def smiles_fits_in_vocab(smi: str):
    """ Check if a SMILES string has unfamiliar tokens.

    :param smiles: SMILES string
    :param extra_patterns: extra tokens to consider (default = None)
        e.g. metalloids: ['Si', 'As', 'Te', 'te', 'B', 'b']  (in ChEMBL33: B+b=0.23%, Si=0.13%, As=0.01%, Te+te=0.01%).
        Mind you that the order matters. If you place 'C' before 'Cl', all Cl tokens will actually be tokenized as C,
        meaning that subsets should always come after superset strings, aka, place two letter elements first in the list
    :return: True if the smiles string has unfamiliar tokens
    """

    tokens = smiles_tokenizer(smi)

    if any([i in smi for i in ['.', '9', '%', '-]', '+]']]):
        return False

    if len(tokens) > VOCAB['max_len'] - 2:
        return False

    if len(''.join(tokens)) != len(smi):
        return False

    return True


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


