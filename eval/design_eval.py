import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from constants import VOCAB
from torch.utils.data.dataloader import default_collate
from dataprep.descriptors import encoding_to_smiles
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Draw import MolToImage
from typing import Union


def draw_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    im = MolToImage(mol)
    plt.imshow(im)
    plt.axis('off')
    plt.show()


def smiles_validity(smiles: list[str]):
    valid_smiles = get_valid_designs(smiles)
    validity = len(valid_smiles) / len(smiles)

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


def get_valid_designs(design_list: list[str]) -> list[str]:
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

    return [design for design in cleaned_designs if design is not None]


def strip_smiles(smiles: list[str]):
    start_token, end_token, padding_token = VOCAB['start_char'], VOCAB['end_char'], VOCAB['pad_char']
    stripped = [smi.replace(start_token, '').replace(end_token, '').replace(padding_token, '') for smi in smiles]

    return stripped










designs = ['COCCN(c1ccccc1)C(C)c1nn(C)cc1O', 'CC1CN(Cc2nnsc2Cl)CC1Nc1nccs1', 'N#Cc1cccc(/C=C/c2ccccc2)c1', 'CCOC1CN(CC2CC2)C2CCCOC12', 'O=C(NCc1ccc(Cl)cc1)c1ccncc1', 'CC1OC(O)C(S)C(O)C1O', 'COc1ccc2oc(=O)c(=O)[nH]c2c1', 'Cc1nc(CNc2cccc(Cl)c2Cl)cs1', 'CCn1c(=O)[nH]c2cc(C)nn2c1=O', 'CC(=CCC(N)Cc1ccccc1)c1nc(C)no1', 'Cc1cc(NCCNN2CCOCC2)n2nccc2n1', 'CN(C)C(=O)c1ccc(OC2CCOCC2)cc1', 'O=C1NC(CO)C(O)C(O)C1O', 'C/C(=N/NC(=S)N1CCNCC1)c1ccccc1', 'O=C(NCCN1CCOCC1)c1cc(Cl)ccc1Cl', 'Nc1cccc(Cl)c1CC=NNc1ccccn1', 'O=C(CCC1CCCCC1)NCCC1CCOCC1', 'NN(CC(=O)O)C(=O)NCCc1ccco1', 'CN1CCOc2nncc(N)c2C1=O', 'COC(=O)NC1=NC(c2ccccc2N)NN1', 'Oc1nc(CSc2nn[nH]n2)nc2ccccc12', 'O=C1NC(=S)NC1=Cc1cccs1', 'COc1ccc(/N=C/c2ccc(O)cc2)cc1', 'Cc1ccc(C(=O)NCc2cccnc2)o1']




