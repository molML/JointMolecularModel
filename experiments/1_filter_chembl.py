"""


Derek van Tilborg
Eindhoven University of Technology
June 2024
"""

import os
from warnings import warn
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from cheminformatics.utils import smiles_to_mols, get_scaffold
from constants import ROOTDIR


def get_all_finetuning_molecules():
    datasets = [i for i in os.listdir('data/clean') if not i.startswith('ChEMBL_33')]

    smiles = []
    for dataset in datasets:
        smiles.extend(pd.read_csv(f'data/clean/{dataset}').smiles)

    return list(set(smiles))


def mols_to_scaffolds(mols):
    scaffolds = []

    for m in tqdm(mols):
        scaffolds.append(get_scaffold(m, 'bemis_murcko'))

    return scaffolds


def filter_unique_mols(mols: list) -> list:
    """ return unique molecules """
    smiles = [Chem.MolToSmiles(m) for m in mols]
    unique_smiles = list(set(smiles))

    return smiles_to_mols(unique_smiles)


if __name__ == '__main__':

    os.chdir(ROOTDIR)

    # Get all fine-tuning data and load ChEMBL data
    finetuning_smiles = get_all_finetuning_molecules()
    chembl33 = pd.read_csv('data/clean/ChEMBL_33.csv').smiles.tolist()#[:5000]

    finetuning_mols = smiles_to_mols(finetuning_smiles)
    chembl33_mols = smiles_to_mols(chembl33)

    # Compute the molecular frameworks
    chembl33_frameworks = mols_to_scaffolds(chembl33_mols)
    chembl33_frameworks_smiles = [Chem.MolToSmiles(m) for m in chembl33_frameworks]

    # Get the unique finetuning frameworks and get rid of the empty '' framework
    finetuning_frameworks = mols_to_scaffolds(finetuning_mols)
    finetuning_frameworks_smiles = [Chem.MolToSmiles(m) for m in finetuning_frameworks]
    finetuning_frameworks_smiles = [smi for smi in finetuning_frameworks_smiles if smi != '']
    finetuning_frameworks_smiles = finetuning_frameworks_smiles

    # Find the matches of molecules in the pre-training set and every fine-tuning molecule. If they are too similar, I
    # will remove these molecules from the pre-training set to make sure that the pretrained model cannot be biased
    # towards
    chembl33_no_overlap = []
    for chembl33_original_smi, chembl33_frameworks_smi in tqdm(zip(chembl33, chembl33_frameworks_smiles)):
        if not chembl33_frameworks_smi in finetuning_frameworks_smiles:
            chembl33_no_overlap.append(chembl33_original_smi)

    # Save to file
    pd.DataFrame({'smiles': chembl33_no_overlap}).to_csv('data/clean/ChEMBL_33_filtered.csv', index=False)
