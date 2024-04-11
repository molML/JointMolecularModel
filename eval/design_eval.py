
from typing import Union
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Draw import MolToImage
from rdkit.DataStructs import TanimotoSimilarity
from Levenshtein import distance as edit_distance
from dataprep.descriptors import mols_to_ecfp
from constants import VOCAB


def uniqueness(designs: list[str]) -> float:
    """  Calculates the ratio of unique designs

    :param designs: list of SMILES strings
    :return: ratio of uniques
    """
    return len(set(designs))/len(designs)


def novelty(designs: list[str], train_set: list[str]) -> float:
    """ Calculates the fraction of unique designs that do not occur in the train set

    :param designs: list of SMILES strings
    :param train_set: train SMILES strings
    :return: ratio of new designs
    """
    novel_designs = [smi for smi in set(designs) if smi not in train_set]

    return len(novel_designs) / len(designs)


def reconstruction(predicted_smiles: str, target_smiles: str) -> int:
    """ checks if a SMILES string has been successfully reconstructed

    :param predicted_smiles: generated SMILES string
    :param target_smiles: target SMILES string
    :return: 1 if reconstructed else 0
    """

    return 1 if predicted_smiles == target_smiles else 0


def reconstruction_edit_distance(predicted_smiles: str, target_smiles: str, normalize: bool = True) -> float:
    """ calculates the edit/Levenshtein distances between two strings

    :param predicted_smiles: generated SMILES string
    :param target_smiles: target SMILES string
    :param normalize: normalizes the edit distance by the target string length (default=True)
    :return: edit distance
    """

    dist = edit_distance(predicted_smiles, target_smiles)

    if normalize:
        dist = dist/len(target_smiles)

    return dist


def reconstruction_tanimoto_similarity(predicted_smiles: str, target_smiles: str, **kwargs) -> float:
    """ calculates the Tanimoto similarity between two SMILES strings using ECFPs (2048 bits, radius 2 by default).
    SMILES strings must be valid molecules!

    :param predicted_smiles: generated SMILES string
    :param target_smiles: target SMILES string
    :param **kwargs: kwargs passed to mols_to_ecfp (e.g., radius=2, nbits=2048)
    :return: Tanimoto similarity
    """
    pred_mol = Chem.MolFromSmiles(predicted_smiles)
    target_mol = Chem.MolFromSmiles(target_smiles)

    fps = mols_to_ecfp([pred_mol, target_mol], **kwargs)

    return TanimotoSimilarity(fps[0], fps[1])


def draw_mol_comparison(smiles1: str, smiles2: str = None):
    """ Plots one or two molecules from their SMILES string """

    im1 = MolToImage(Chem.MolFromSmiles(smiles1))
    if smiles2 is None:
        f, axarr = plt.subplots(1, 1, figsize=(3, 3))
        axarr.imshow(im1)
        axarr.axis('off')
    else:
        im2 = MolToImage(Chem.MolFromSmiles(smiles2))
        f, axarr = plt.subplots(2, 1, figsize=(3, 6))
        axarr[0].imshow(im1)
        axarr[0].axis('off')
        axarr[1].imshow(im2)
        axarr[1].axis('off')

    plt.show()


def smiles_validity(smiles: list[str], return_invalids: bool = False) -> (float, list):
    """ Checks SMILES validity over a list of SMILES strings

    :param smiles: list of plausible SMILES strings
    :param return_invalids: if True, returns None for invalids, if False (default), only returns valids
    :return: ratio of valid molecules, list of SMILES
    """
    valid_smiles = get_valid_designs(smiles, return_invalids)
    validity = len([smi for smi in valid_smiles if smi is not None]) / len(smiles)

    return validity, valid_smiles


def clean_design(smi: str) -> Union[str, None]:
    """
    Cleans a given SMILES string by performing the following steps:
    1. Converts the SMILES string to a molecule object using RDKit.
    2. Removes any charges from the molecule.
    3. Sanitizes the molecule by checking for any errors or inconsistencies.
    4. Converts the sanitized molecule back to a canonical SMILES string.
    Parameters
    ----------
    smi: str
        A SMILES design that possibly represents a chemical compound.
    Returns
    -------
    str
        A cleaned and canonicalized SMILES string representing a chemical compound.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    uncharger = rdMolStandardize.Uncharger()
    mol = uncharger.uncharge(mol)
    sanitization_flag = Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL, catchErrors=True)

    # SANITIZE_NONE is the "no error" flag of rdkit!
    if sanitization_flag != Chem.SanitizeFlags.SANITIZE_NONE:
        return None

    can_smiles = Chem.MolToSmiles(mol, canonical=True)
    if can_smiles is None or len(can_smiles) == 0:
        return None

    return can_smiles


def get_valid_designs(design_list: list[str], return_invalids: bool = False) -> list[str]:
    """
    Filters a list of SMILES strings to only keep the valid ones.
    Applies the `clean_design` function to each SMILES string in the list.
    So, uncharging, sanitization, and canonicalization are performed on each SMILES string.
    Parameters
    ----------
    design_list : List[str]
        A list of SMILES designs representing chemical compounds.
    Returns
    -------
    List[str]
        A list of valid SMILES strings representing chemical compounds.
    """
    RDLogger.DisableLog('rdApp.*')
    cleaned_designs = [clean_design(design) for design in design_list]
    RDLogger.EnableLog('rdApp.*')
    if return_invalids:
        return cleaned_designs

    return [design for design in cleaned_designs if design is not None]


def strip_smiles(smiles: list[str]):
    start_token, end_token, padding_token = VOCAB['start_char'], VOCAB['end_char'], VOCAB['pad_char']
    stripped = [smi.replace(start_token, '').replace(end_token, '').replace(padding_token, '') for smi in smiles]

    return stripped
