"""
Code to clean up SMILES strings

- clean_mols: Cleans up a molecule
- has_unfamiliar_tokens: Check if a SMILES string has unfamiliar tokens
- flatten_stereochemistry: Get rid of stereochemistry in a SMILES string
- desalter: Get rid of salt from SMILES strings
- remove_common_solvents: Get rid of some of the most commonly used solvents in a SMILES string
- unrepeat_smiles: If a SMILES string contains repeats of the same molecule, return a single one of them
- sanitize_mols: Sanitize a molecules with RDkit
- neutralize_mols: Use pre-defined reactions to neutralize charged molecules

Derek van Tilborg
Eindhoven University of Technology
Jan 2024
"""
from typing import Union
import math
import warnings
from tqdm import tqdm
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit import Chem
from rdkit.Chem import AllChem
from cheminformatics.utils import canonicalize_smiles, smiles_tokenizer, smiles_to_mols
from cheminformatics.descriptors import mols_to_ecfp
from constants import VOCAB


SOLVENTS = ['O=C(O)C(F)(F)F', 'O=C(O)C(=O)O', 'O=C(O)/C=C/C(=O)O', 'CS(=O)(=O)O', 'O=C(O)/C=C\\C(=O)O', 'CC(=O)O',
            'O=S(=O)(O)O', 'O=CO', 'CCN(CC)CC', '[O-][Cl+3]([O-])([O-])[O-]', 'O=C(O)C(O)C(O)C(=O)O',
            'Cc1ccc(S(=O)(=O)[O-])cc1', 'O=C([O-])C(F)(F)F', 'Cc1ccc(S(=O)(=O)O)cc1', 'O=C(O)CC(O)(CC(=O)O)C(=O)O',
            'O=[N+]([O-])O', 'F[B-](F)(F)F', 'O=S(=O)([O-])C(F)(F)F', 'F[P-](F)(F)(F)(F)F', 'O=C(O)CCC(=O)O',
            'O=P(O)(O)O', 'NCCO', 'CS(=O)(=O)[O-]', '[O-][Cl+3]([O-])([O-])O', 'COS(=O)(=O)[O-]', 'NC(CO)(CO)CO',
            'CCO', 'CN(C)C=O', 'O=C(O)[C@H](O)[C@@H](O)C(=O)O', 'C1CCC(NC2CCCCC2)CC1', 'C', 'O=S(=O)([O-])O',
            'CNC[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO', 'c1ccncc1']

NEUTRALIZATION_PATTERNS = (
        # Imidazoles
        ('[n+;H]', 'n'),
        # Amines
        ('[N+;!H0]', 'N'),
        # Carboxylic acids and alcohols
        ('[$([O-]);!$([O-][#7])]', 'O'),
        # Thiols
        ('[S-;X1]', 'S'),
        # Sulfonamides
        ('[$([N-;X2]S(=O)=O)]', 'N'),
        # Enamines
        ('[$([N-;X2][C,N]=C)]', 'N'),
        # Tetrazoles
        ('[n-]', '[nH]'),
        # Sulfoxides
        ('[$([S-]=O)]', 'S'),
        # Amides
        ('[$([N-]C=O)]', 'N'),
    )

ISOTOPES = ['[11c]', '[14C]', '[10B]', '[11C]', '[15n]', '[14c]', '[17F]', '[3H]', '[18F]', '[13C]', '[19F]', '[18O]',
            '[2H]']


def clean_mols(smiles: list[str]) -> (dict, dict):
    """ Cleans SMILES strings in the following steps:
    1. flattening stereochemistry
    2. desalting
    3. removing common solvents
    4. removing duplicated fragments
    5. sanitization of the mol object
    6. neutralization
    7. canonicalization
    8. checking for uncommon SMILES characters
    9. checking if the molecule can be featurized into an ECFP

    :param smiles: SMILES strings that are in need for cleaning
    :return: two dicts: cleaned_smiles and failed_smiles, containing the original SMILES strings and the cleaned/failed
    """

    cleaned_smiles = {'original': [], 'clean': []}
    failed_smiles = {'original': [], 'reason': []}

    for smi in tqdm(smiles):
        try:
            smi_clean, reason = clean_single_mol(smi)
            if smi_clean:
                cleaned_smiles['original'].append(smi)
                cleaned_smiles['clean'].append(smi_clean)
            else:
                failed_smiles['original'].append(smi)
                failed_smiles['reason'].append(reason)
        except:
            failed_smiles['original'].append(smi)
            failed_smiles['reason'].append('Other')

    return cleaned_smiles, failed_smiles


def clean_single_mol(smi):

    smi = flatten_stereochemistry(smi)
    smi = desalter(smi)
    smi = remove_common_solvents(smi)
    smi = unrepeat_smiles(smi)
    smi = sanitize_mol(smi)
    smi = neutralize_mol(smi)
    smi = canonicalize_smiles(smi)

    if not type(smi) is str or smi is None:
        return None, 'Other'

    if type(smi) is float:
        return None, 'Nan'

    # Remove giant ring systems, fragmented molecules, and charged molecules
    if not smiles_fits_in_vocab(smi):
        return None, 'Does not fit vocab'

    if any([i in smi for i in ISOTOPES]):
        return None, 'Isotope'

    if not mols_to_ecfp(smiles_to_mols(smi)):
        return None, 'Featurization'

    return smi, None


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


def flatten_stereochemistry(smiles: str) -> str:
    """ Remove stereochemistry from a SMILES string """
    return smiles.replace('@', '')


def desalter(smiles, salt_smarts: str = "[Cl,Na,Mg,Ca,K,Br,Zn,Ag,Al,Li,I,O,N,H]") -> str:
    """ Get rid of salt from SMILES strings, e.g., CCCCCCCCC(O)CCC(=O)[O-].[Na+] -> CCCCCCCCC(O)CCC(=O)[O-]

    :param smiles: SMILES string
    :param salt_smarts: SMARTS pattern to remove all salts (default = "[Cl,Br,Na,Zn,Mg,Ag,Al,Ca,Li,I,O,N,K,H]")
    :return: cleaned SMILES w/o salts
    """
    if '.' not in smiles:
        return smiles

    remover = SaltRemover(defnData=salt_smarts)

    new_smi = Chem.MolToSmiles(remover.StripMol(Chem.MolFromSmiles(smiles)))

    return new_smi


def remove_common_solvents(smiles: str) -> str:
    """ Remove commonly used solvents from a SMILES string, e.g.,
    Nc1ncnc2scc(-c3ccc(NC(=O)Cc4cc(F)ccc4F)cc3)c12.O=C(O)C(F)(F)F -> Nc1ncnc2scc(-c3ccc(NC(=O)Cc4cc(F)ccc4F)cc3)c12

     The following solvents are removed:

    'O=C(O)C(F)(F)F', 'O=C(O)C(=O)O', 'O=C(O)/C=C/C(=O)O', 'CS(=O)(=O)O', 'O=C(O)/C=C\\C(=O)O', 'CC(=O)O',
    'O=S(=O)(O)O', 'O=CO', 'CCN(CC)CC', '[O-][Cl+3]([O-])([O-])[O-]', 'O=C(O)C(O)C(O)C(=O)O',
    'Cc1ccc(S(=O)(=O)[O-])cc1', 'O=C([O-])C(F)(F)F', 'Cc1ccc(S(=O)(=O)O)cc1', 'O=C(O)CC(O)(CC(=O)O)C(=O)O',
    'O=[N+]([O-])O', 'F[B-](F)(F)F', 'O=S(=O)([O-])C(F)(F)F', 'F[P-](F)(F)(F)(F)F', 'O=C(O)CCC(=O)O', 'O=P(O)(O)O',
    'NCCO', 'CS(=O)(=O)[O-]', '[O-][Cl+3]([O-])([O-])O', 'COS(=O)(=O)[O-]', 'NC(CO)(CO)CO', 'CCO', 'CN(C)C=O',
    'O=C(O)[C@H](O)[C@@H](O)C(=O)O', 'C1CCC(NC2CCCCC2)CC1', 'C', 'O=S(=O)([O-])O',
    'CNC[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO', 'c1ccncc1'

     (not the most efficient code out there)
    :param smiles: SMILES string
    :return: cleaned SMILES
    """
    if '.' not in smiles:
        return smiles

    for solv in SOLVENTS:
        smiles = desalter(smiles, solv)

    return smiles


def unrepeat_smiles(smiles: str) -> str:
    """ if a SMILES string contains repeats of the same molecule, return a single one of them

    :param smiles: SMILES string
    :return: unrepeated SMILES string if repeats were found, else the original SMILES string
    """
    if '.' not in smiles:
        return smiles

    repeats = set(smiles.split('.'))
    if len(repeats) > 1:
        return smiles
    return list(repeats)[0]


def _initialise_neutralisation_reactions() -> list[(str, str)]:
    """ adapted from the rdkit contribution of Hans de Winter """
    return [(Chem.MolFromSmarts(x), Chem.MolFromSmiles(y, False)) for x, y in NEUTRALIZATION_PATTERNS]


def sanitize_mol(smiles: str) -> Union[str, None]:
    """ Sanitize a molecules with RDkit

    :param smiles: SMILES string
    :return: SMILES string if sanitized or None if failed sanitizing
    """
    # basic checks on SMILES validity
    mol = Chem.MolFromSmiles(smiles)

    # flags: Kekulize, check valencies, set aromaticity, conjugation and hybridization
    san_opt = Chem.SanitizeFlags.SANITIZE_ALL

    if mol is not None:
        sanitize_error = Chem.SanitizeMol(mol, catchErrors=True, sanitizeOps=san_opt)
        if sanitize_error:
            warnings.warn(sanitize_error)
            return None
    else:
        return None

    return Chem.MolToSmiles(mol)


def neutralize_mol(smiles: str) -> str:
    """ Use several neutralisation reactions based on patterns defined in NEUTRALIZATION_PATTERNS to neutralize charged
    molecules

    :param smiles: SMILES string
    :return: SMILES of the neutralized molecule
    """
    mol = Chem.MolFromSmiles(smiles)

    # retrieves the transformations
    transfm = _initialise_neutralisation_reactions()  # set of transformations

    # applies the transformations
    for i, (reactant, product) in enumerate(transfm):
        while mol.HasSubstructMatch(reactant):
            rms = AllChem.ReplaceSubstructs(mol, reactant, product)
            mol = rms[0]

    # converts back the molecule to smiles
    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

    return smiles


def randomize_smiles_string(smi: str) -> str:
    """ Randomize a SMILES string. Check if the new SMILES is not out of bounds with our vocab rules

    :param smi: SMILES string
    :return: randomized SMILES string
    """
    random_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=False, doRandom=True)

    if smiles_fits_in_vocab(random_smi):
        return random_smi

    return smi
