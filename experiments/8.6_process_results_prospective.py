""" Rank molecules for prospective screening using utopia point distance with uncertainty and unfamiliarity
(+ expected value of course, we only want molecules that are predicted as hits).

For each protein target, PIM1 (CHEMBL2147_Ki), CDK1 (CHEMBL308_Ki), and MNK1 (CHEMBL4718_Ki), we select the top
molecules predicted as hits according to:
- Least uncertain + Least unfamiliar
- Least uncertain + Most unfamiliar
- Most uncertain + Least unfamiliar

We skip over all molecules that are too similar to molecules we already selected or too similar to the training data.
Molecules most adhere to some basic 'kinase inhibitor requirements'.

Derek van Tilborg
Eindhoven University of Technology
April 2025
"""

import os
from os.path import join as ospj
from itertools import batched
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from cheminformatics.multiprocessing import tanimoto_matrix
from cheminformatics.utils import smiles_to_mols
from cheminformatics.kinase_filters import find_kinase_violations
from constants import ROOTDIR


def get_all_ecfps(smiles: list[str], path: str = None):

    if os.path.exists(path):
        print(f'Loading precomputed ECFPs from {path}')
        all_mol_info = torch.load(path)
    else:
        all_smiles = list(set(smiles))

        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

        all_mol_info = {}
        for smi in tqdm(all_smiles):
            try:
                mol = smiles_to_mols(smi)
                ecfp = mfpgen.GetFingerprint(mol)
                all_mol_info[smi] = {'mol': mol, 'ecfp': ecfp}
            except:
                print(f"failed {smi}")

        torch.save(all_mol_info, path)

    return all_mol_info


def compute_dataset_library_distance(all_library_smiles, dataset_name, all_mol_info):

    # get the SMILES from the pre-processed file
    data_path = ospj(f'data/clean/{dataset_name}.csv')
    df = pd.read_csv(data_path)
    dataset_smiles = df.smiles.tolist()

    # if there are any failed smiles, remove them
    dataset_smiles = [smi for smi in dataset_smiles if smi in all_mol_info]
    all_library_smiles = [smi for smi in all_library_smiles if smi in all_mol_info]

    # chunk all smiles in batches for efficient multi-threading with manageable memory
    all_library_smiles_batches = [i for i in batched(all_library_smiles, 10000)]

    # tanimoto sim between every screening mol and the training mols
    train_ecfps = [all_mol_info[smi]['ecfp'] for smi in dataset_smiles]
    ECFPs_S_mean = []
    ECFPs_S_max = []
    for batch in tqdm(all_library_smiles_batches, desc='\tComputing Tanimoto similarity', unit_scale=10000):
        T_to_train = tanimoto_matrix([all_mol_info[smi]['ecfp'] for smi in batch],
                                     train_ecfps, take_mean=False)
        ECFPs_S_mean.append(np.mean(T_to_train, 1))
        ECFPs_S_max.append(np.max(T_to_train, 1))

    ECFPs_S_mean = np.concatenate(ECFPs_S_mean)
    ECFPs_S_max = np.concatenate(ECFPs_S_max)

    df = {'smiles': all_library_smiles,
          'dataset': dataset_name,
          'Tanimoto_to_dataset_mean': ECFPs_S_mean,
          'Tanimoto_to_dataset_max': ECFPs_S_max
          }

    return pd.DataFrame(df)

def average_over_seed(df):
    # function to average over seeds: take the mean of numeric values, take the most common string value (strings
    # should be all the same though)
    agg_funcs = {
        col: 'mean' if pd.api.types.is_numeric_dtype(df[col]) else (
            lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA)
        for col in df.columns if col not in ['smiles', 'split']
    }

    # Group by two columns and aggregate
    df = df.groupby(['smiles', 'split']).agg(agg_funcs).reset_index()
    df = df.drop('seed', axis=1)

    return df


if __name__ == '__main__':

    os.chdir(ROOTDIR)

    # filter out molecules too similar to the train (and to each other during ranking)
    TANIMOTO_CUTOFF = 0.7

    # Select the top 10 ranked molecules
    TOP_N = 10

    DATA_DIR = os.path.join('results', 'prospective')
    SPECS_PATH = "data/screening_libraries/specs_2025/specs_clean_Apr2025.csv"
    DATASETS = ['CHEMBL4718_Ki', 'CHEMBL308_Ki', 'CHEMBL2147_Ki']

    # Load screening results
    screening_results = []
    for dataset in DATASETS:
        df = pd.read_csv(os.path.join(DATA_DIR, 'smiles_jmm', dataset, 'results_preds.csv'))

        # take the mean over the 10 different seeds
        df = average_over_seed(df)

        # rename some columns and remove columns we don't need
        df['dataset'] = dataset
        df['library'] = 'specs_Apr2025'
        df = df[['smiles', 'split', 'reconstruction_loss', 'edit_distance', 'y_hat', 'y_unc', 'y_E', 'library', 'dataset']]
        df = df.rename(columns={'reconstruction_loss': 'unfamiliarity'})
        screening_results.append(df)
    screening_results = pd.concat(screening_results)

    # Get all SMILES (datasets + library) and precompute ecfps
    all_dataset_smiles = list(
        set(sum([pd.read_csv(ospj(f'data/clean/{ds}.csv'))['smiles'].tolist() for ds in DATASETS], [])))
    all_smiles = list(set(screening_results['smiles'].tolist() + all_dataset_smiles))
    all_ecfps = get_all_ecfps(all_smiles, ospj('data', 'screening_libraries', 'specs_2025', 'ecfps.pt'))

    # Compute all distances and combine them with the inference results
    dataset_distances = []
    for dataset_name in DATASETS:
        dataset_distances.append(compute_dataset_library_distance(all_smiles, dataset_name, all_ecfps))
    dataset_distances = pd.concat(dataset_distances)
    screening_results = screening_results.merge(dataset_distances, how="left", on=["smiles", "dataset"])

    # Calculate kinase-like characteristics of all molecules. We later use this to triage molecules
    screening_results['kinase_violations'] = [find_kinase_violations(smi) for smi in tqdm(screening_results['smiles'])]

    # Get the original Specs ID and combine them with the inference results
    specs_library = pd.read_csv(SPECS_PATH)
    specs_library = specs_library[['specs_ID', 'smiles_original', 'smiles_cleaned', 'url']]
    specs_library = specs_library.rename(columns={'smiles_cleaned': 'smiles', 'smiles_original': 'specs_smiles'})
    screening_results = screening_results.merge(specs_library, how="left", on=["smiles"])

    # write to file
    screening_results.to_csv(os.path.join(DATA_DIR, 'smiles_jmm', 'screening_results_combined.csv'), index=False)
