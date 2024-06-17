"""
Script to clean the data of all finetuning sets from Lit-PCBA, MoleculeACE, and Ames mutagenicity and the pre-training
data from ChEBML33

Derek van Tilborg
June 2024
Eindhoven University of Technology
"""

import os
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from rdkit.DataStructs import BulkTanimotoSimilarity
from cheminformatics.cleaning import clean_mols
from cheminformatics.utils import smiles_to_mols, mols_to_scaffolds, mols_to_smiles
from cheminformatics.splitting import scaffold_split, random_split
from cheminformatics.descriptors import mols_to_ecfp


def process_litpcba(dataset_name: str) -> pd.DataFrame:

    # read data
    with open(f'data/LitPCBA/{dataset_name}/actives.smi', 'r') as file:
        actives = [line.strip().split(' ')[0] for line in file]
    with open(f'data/LitPCBA/{dataset_name}/inactives.smi', 'r') as file:
        inactives = [line.strip().split(' ')[0] for line in file]

    # Clean molecules
    actives, actives_failed = clean_mols(actives)
    inactives, inactives_failed = clean_mols(inactives)

    print('Parsing errors:')
    [print(f"{k}: {v}") for k, v in Counter(actives_failed['reason'] + inactives_failed['reason']).items()]

    # Get the sets
    actives_clean = set(actives['clean'])
    inactives_clean = set(inactives['clean'])

    # Check if there are overlapping molecules in both the set of inactives and actives
    intersecting_smiles = actives_clean & inactives_clean
    actives_clean = [smi for smi in actives_clean if smi not in intersecting_smiles]
    inactives_clean = [smi for smi in inactives_clean if smi not in intersecting_smiles]

    # Put it all together
    y = [1] * len(actives_clean) + [0] * len(inactives_clean)
    smiles = list(actives_clean) + list(inactives_clean)

    # put together in a dataframe and shuffle the rows
    df = pd.DataFrame({'smiles': smiles, 'y': y})
    df = df.sample(frac=1).reset_index(drop=True)

    df.to_csv(f'data/clean/{dataset_name}.csv')


def process_moleculeace(dataset_name: str) -> pd.DataFrame:
    activity_threshold = 100  # in nM

    df_original = pd.read_csv(f'data/moleculeace_original/{dataset_name}.csv')

    actives = df_original.loc[df_original['exp_mean [nM]'] <= activity_threshold, 'smiles'].tolist()
    inactives =  df_original.loc[df_original['exp_mean [nM]'] > activity_threshold, 'smiles'].tolist()

    # Clean molecules
    actives, actives_failed = clean_mols(actives)
    inactives, inactives_failed = clean_mols(inactives)

    print('Parsing errors:')
    [print(f"{k}: {v}") for k, v in Counter(actives_failed['reason'] + inactives_failed['reason']).items()]

    # Get the sets
    actives_clean = set(actives['clean'])
    inactives_clean = set(inactives['clean'])

    # Check if there are overlapping molecules in both the set of inactives and actives
    intersecting_smiles = actives_clean & inactives_clean
    actives_clean = [smi for smi in actives_clean if smi not in intersecting_smiles]
    inactives_clean = [smi for smi in inactives_clean if smi not in intersecting_smiles]

    # Put it all together
    y = [1] * len(actives_clean) + [0] * len(inactives_clean)
    smiles = list(actives_clean) + list(inactives_clean)

    # put together in a dataframe and shuffle the rows
    df = pd.DataFrame({'smiles': smiles, 'y': y})
    df = df.sample(frac=1).reset_index(drop=True)

    df.to_csv(f'data/clean/{dataset_name}.csv')


def process_ames():

    actives, inactives = [], []
    with open('data/Ames_mutagenicity/smiles_cas_N6512.smi', 'r') as file:
        for line in file:
            label = int(line.strip().split('\t')[-1])
            smi = line.strip().split(' ')[0]
            if label == 1:
                actives.append(smi)
            else:
                inactives.append(smi)

    # Clean molecules
    actives, actives_failed = clean_mols(actives)
    inactives, inactives_failed = clean_mols(inactives)

    print('Parsing errors:')
    [print(f"{k}: {v}") for k, v in Counter(actives_failed['reason'] + inactives_failed['reason']).items()]

    # Get the sets
    actives_clean = set(actives['clean'])
    inactives_clean = set(inactives['clean'])

    # Check if there are overlapping molecules in both the set of inactives and actives
    intersecting_smiles = actives_clean & inactives_clean
    actives_clean = [smi for smi in actives_clean if smi not in intersecting_smiles]
    inactives_clean = [smi for smi in inactives_clean if smi not in intersecting_smiles]

    # Put it all together
    y = [1] * len(actives_clean) + [0] * len(inactives_clean)
    smiles = list(actives_clean) + list(inactives_clean)

    # put together in a dataframe and shuffle the rows
    df = pd.DataFrame({'smiles': smiles, 'y': y})
    df = df.sample(frac=1).reset_index(drop=True)

    df.to_csv(f'data/clean/Ames_mutagenicity.csv')


def process_chembl():

    # Read ChEMBL 33
    chembl_smiles = pd.read_table("data/ChEMBL/chembl_33_chemreps.txt").canonical_smiles.tolist()

    print('started with', len(chembl_smiles))  # 2,372,674
    # Clean smiles and get rid of duplicates
    chembl_smiles_clean, chembl_smiles_failed = clean_mols(chembl_smiles)

    print('Parsing errors:')
    [print(f"{k}: {v}") for k, v in Counter(chembl_smiles_failed['reason']).items()]

    print('clean smiles', len(chembl_smiles_clean['clean']))

    chembl_smiles_clean = list(set(chembl_smiles_clean['clean']))
    chembl_smiles_clean = [smi for smi in chembl_smiles_clean if type(smi) is str and smi != '']
    '''1,974,867 were unique '''
    print('uniques', len(chembl_smiles_clean))

    # Save cleaned SMILES strings to a csv file for later use
    pd.DataFrame({'smiles': chembl_smiles_clean}).to_csv("data/clean/ChEMBL_33.csv", index=False)


if __name__ == '__main__':

    # LitPCBA datasets that are not giant and have a decent active/inactive ratio: 'ESR1_ant', 'TP53', 'PPARG'
    process_litpcba('ESR1_ant')
    process_litpcba('TP53')
    process_litpcba('PPARG')

    # MoleculeACE datasets
    process_moleculeace('CHEMBL4203_Ki')
    process_moleculeace('CHEMBL2034_Ki')
    process_moleculeace('CHEMBL233_Ki')
    process_moleculeace('CHEMBL4616_EC50')
    process_moleculeace('CHEMBL287_Ki')
    process_moleculeace('CHEMBL218_EC50')
    process_moleculeace('CHEMBL264_Ki')
    process_moleculeace('CHEMBL219_Ki')
    process_moleculeace('CHEMBL2835_Ki')
    process_moleculeace('CHEMBL2147_Ki')
    process_moleculeace('CHEMBL231_Ki')
    process_moleculeace('CHEMBL3979_EC50')
    process_moleculeace('CHEMBL237_EC50')
    process_moleculeace('CHEMBL244_Ki')
    process_moleculeace('CHEMBL4792_Ki')
    process_moleculeace('CHEMBL1871_Ki')
    process_moleculeace('CHEMBL237_Ki')
    process_moleculeace('CHEMBL262_Ki')
    process_moleculeace('CHEMBL2047_EC50')
    process_moleculeace('CHEMBL2971_Ki')
    process_moleculeace('CHEMBL237_Ki')
    process_moleculeace('CHEMBL204_Ki')
    process_moleculeace('CHEMBL214_Ki')
    process_moleculeace('CHEMBL1862_Ki')
    process_moleculeace('CHEMBL234_Ki')
    process_moleculeace('CHEMBL238_Ki')
    process_moleculeace('CHEMBL235_EC50')
    process_moleculeace('CHEMBL4005_Ki')
    process_moleculeace('CHEMBL236_Ki')
    process_moleculeace('CHEMBL228_Ki')

    # Ames Mutagenicity dataset
    process_ames()  # many molecules in this dataset are either charged or invalid SMILES strings

    # The pre-training data: ChEMBL33
    process_chembl()

        # Parsing errors:
        # Does not fit vocab: 231,632
        # Isotope: 1,516
        # None: 53
        # Other: 1
        # clean smiles 2,139,472
        # uniques 1,974,867
