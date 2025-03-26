"""
Filter the pretraining molecules by removing all scaffolds similar to the finetuning molecules

Derek van Tilborg
Eindhoven University of Technology
March 2024
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join as ospj
from rdkit import Chem
from rdkit.DataStructs import BulkTanimotoSimilarity
from cheminformatics.utils import smiles_to_mols, get_scaffold
from cheminformatics.descriptors import mols_to_ecfp
from cheminformatics.cleaning import clean_mols
from constants import ROOTDIR

SIM_THRESHOLD = 0.7
SCAFFOLD_TYPE = 'bemis_murcko'
RADIUS = 2
NBITS = 2048


def split_chembl(df: pd.DataFrame, random_state: int = 1) -> pd.DataFrame:
    """ shuffle randomly and add a column with train (0.8), test (0.1), and val (0.1)

    :param df: dataframe with a `smiles` column containing SMILES strings
    :param random_state: seed
    :param test_frac: fraction of molecules that goes to the test set
    :param val_frac: fraction of molecules that goes to the val set
    :return: dataframe with a `smiles` columns and a `split` column
    """

    # shuffle dataset
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # create splits and add them as a column
    split_col = (['val'] * int(len(df) * 0.1))
    split_col = (['train'] * (len(df) - len(split_col))) + split_col
    df['split'] = split_col

    return df


def get_all_finetuning_molecules() -> list[str]:
    """ Go through all finetuning datasets (ie everything but ChEMBL) and return the unique SMILES """

    datasets = ['CHEMBL4718_Ki', 'CHEMBL308_Ki', 'CHEMBL2147_Ki']

    smiles = []
    for dataset in datasets:
        smiles.extend(pd.read_csv(f'data/clean/{dataset}.csv').smiles)

    # canonicalize just to be sure
    smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in smiles]

    return list(set(smiles))


def mols_to_scaffolds(mols: list, scaffold_type: 'bemis_murcko') -> list:
    """ Compute the scaffolds of a list of molecules """

    scaffolds = []

    for m in tqdm(mols):
        scaffolds.append(get_scaffold(m, scaffold_type=scaffold_type))

    return scaffolds


def filter_unique_mols(mols: list) -> list:
    """ return unique molecules """

    smiles = [Chem.MolToSmiles(m) for m in mols]
    unique_smiles = list(set(smiles))

    return smiles_to_mols(unique_smiles)


def get_unique_scaffold_ecfps(smiles: list[str]):
    """ Get all unique scaffold ECFPs from a list of SMILES """

    # Compute all unique scaffolds
    mols = smiles_to_mols(smiles)
    scaffolds = mols_to_scaffolds(mols, scaffold_type=SCAFFOLD_TYPE)
    unique_scaffolds = filter_unique_mols(scaffolds)

    # Get the ECFP fingerprint of all scaffolds
    scaffolds_ecfps = mols_to_ecfp(unique_scaffolds, radius=RADIUS, nbits=NBITS)

    return scaffolds_ecfps


def filter_pretraining_smiles(pretraining_smiles: list[str], finetuning_smiles: list[str]) -> (list[str], list[str]):
    """ Find which SMILES from the pretraining set have a similar scaffold to any of the scaffolds in the finetuning
    sets

    :param pretraining_smiles: all SMILES in the pretraining set
    :param finetuning_smiles: all SMILES in the finetuning set

    :return: SMILES that passed the filter, SMILES that are discarded
    """

    # get the ecfps of all unique finetuning scaffolds
    print("Computing ECFP of all unique finetuning scaffolds")
    finetuning_scaffolds_ecfps = get_unique_scaffold_ecfps(finetuning_smiles)

    # go through all pretraining molecules. If a molecule is too similar to any of the finetuning molecules, drop it
    print("Looking for similar scaffolds in the pretraining data")
    passed, discarded = [], []
    for smi in tqdm(pretraining_smiles):

        scaffold = get_scaffold(smiles_to_mols(smi), scaffold_type=SCAFFOLD_TYPE)
        scaffold_ecfp = mols_to_ecfp(scaffold, radius=RADIUS, nbits=NBITS)
        # Calculate the Tanimoto similarity of this ChEMBL scaffold to all finetuning scaffolds
        sim = BulkTanimotoSimilarity(scaffold_ecfp, finetuning_scaffolds_ecfps)

        # Get rid of empty scaffolds in the pretraining data
        if Chem.MolToSmiles(scaffold) == '':
            discarded.append(smi)

        # If a scaffold is too similar to the finetuning scaffolds, get rid of it
        if np.max(sim) >= SIM_THRESHOLD:
            discarded.append(smi)
        else:
            passed.append(smi)

    return passed, discarded


if __name__ == '__main__':

    OUT_DIR_PATH = 'data/split'
    os.chdir(ROOTDIR)

    # Get all fine-tuning data and load ChEMBL data
    finetuning_smiles = get_all_finetuning_molecules()
    pretraining_smiles = pd.read_csv('data/clean/ChEMBL_33.csv').smiles.tolist()

    # double check for vocab issues
    finetuning_smiles_clean, finetuning_smiles_failed = clean_mols(finetuning_smiles)
    chembl_smiles_clean, chembl_smiles_failed = clean_mols(pretraining_smiles)

    # remove all ChEMBL smiles that are too similar in their scaffold to the finetuning smiles
    chembl33_passed, chembl33_discarded = filter_pretraining_smiles(chembl_smiles_clean['clean'], finetuning_smiles_clean['clean'])

    print(f"{len(chembl33_passed)} passed\n{len(chembl33_discarded)} discarded")

    # Save to file
    pd.DataFrame({'smiles': chembl33_passed}).to_csv('data/clean/ChEMBL_33_filtered_prospective.csv', index=False)
    pd.DataFrame({'smiles': chembl33_discarded}).to_csv('data/clean/ChEMBL_33_filtered_prospective_discarded.csv', index=False)

    # 1,867,127 passed
    # 98,740 discarded

    chembl = split_chembl(pd.DataFrame({'smiles': chembl33_passed}))

    chembl.to_csv(ospj(OUT_DIR_PATH, 'ChEMBL_33_split_prospective.csv'), index=False)
