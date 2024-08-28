
from typing import Union
from Levenshtein import distance as edit_distance
from rdkit import Chem, RDLogger
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D, rdDepictor
from cheminformatics.descriptors import mols_to_ecfp
rdDepictor.SetPreferCoordGen(True)


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

    dist = edit_distance(target_smiles, predicted_smiles)

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


def plot_molecules_acs1996_grid(mols: list[Chem.rdchem.Mol], subImgSize=(300, 250), molsPerRow: int = 4,
                                useSVG: bool = False, legends: list = None):

    opts = rdMolDraw2D.MolDrawOptions()
    # To get the mean bond length, we need coords for the first molecule.
    AllChem.Compute2DCoords(mols[0])
    Draw.SetACS1996Mode(opts, Draw.MeanBondLength(mols[0]))
    img = Draw.MolsToGridImage(mols, molsPerRow=molsPerRow, subImgSize=subImgSize, drawOptions=opts, useSVG=useSVG,
                               legends=legends)

    return img


def plot_molecular_reconstruction(mols_a: list[Chem.rdchem.Mol], mols_b: list[Chem.rdchem.Mol], subImgSize=(300, 250),
                                  labels: list[str] = None):

    assert len(mols_a) == len(mols_b)
    mols = mols_a + mols_b
    labels = [str(i) for i in labels] + [""] * len(mols_b)

    img = plot_molecules_acs1996_grid(mols, subImgSize=subImgSize, molsPerRow=len(mols_a), legends=labels)

    return img


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
