""" Perform model inference for the jcm model

Derek van Tilborg
Eindhoven University of Technology
November 2024
"""

import os
import sys
from os.path import join as ospj
import pandas as pd
from jcm.training_logistics import get_all_dataset_names, load_dataset_df
from constants import ROOTDIR
from jcm.datasets import MoleculeDataset
import torch
from tqdm.auto import tqdm
from cheminformatics.multiprocessing import tanimoto_matrix, bulk_mcsf
from cheminformatics.utils import smiles_to_mols, get_scaffold
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics.pairwise import cosine_similarity
from cheminformatics.descriptors import cats
import numpy as np


def get_all_train_smiles():
    train_smiles = []
    for dataset in get_all_dataset_names():
        df = load_dataset_df(dataset)
        train_smiles.extend(df[df['split'] == 'train'].smiles.tolist())

    return list(set(train_smiles))


def get_all_mol_info(all_library_smiles, all_training_smiles):

    all_mol_info_path = ospj(output_PATH, 'all_mol_info.pt')

    if os.path.exists(all_mol_info_path):
        all_mol_info = torch.load(all_mol_info_path)
    else:
        all_smiles = list(set(all_training_smiles + all_library_smiles))

        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

        all_mol_info = {}
        for smi in tqdm(all_smiles):
            mol = smiles_to_mols(smi)
            scaffold_mol = get_scaffold(mol, scaffold_type='cyclic_skeleton')
            ecfp = mfpgen.GetFingerprint(mol)
            ecfp_scaffold = mfpgen.GetFingerprint(scaffold_mol)
            cats_desriptor = cats(mol)

            all_mol_info[smi] = {'mol': mol, 'ecfp': ecfp, 'ecfp_scaffold': ecfp_scaffold, 'cats': cats_desriptor}

        torch.save(all_mol_info, all_mol_info_path)

    return all_mol_info


def compute_train_library_distance(all_library_smiles, dataset_name, all_mol_info):

    df = load_dataset_df(dataset_name)
    train_smiles = df[df['split'] == 'train'].smiles.tolist()

    # tanimoto sim between every screening mol and the training mols
    print('\tComputing Tanimoto matrix', file=sys.stderr)
    T_to_train = tanimoto_matrix([all_mol_info[smi]['ecfp'] for smi in all_library_smiles],
                                 [all_mol_info[smi]['ecfp'] for smi in train_smiles],
                                 take_mean=True)

    # tanimoto scaffold sim between every screening mol and the training mols
    print('\tComputing scaffold Tanimoto matrix', file=sys.stderr)
    T_scaff_to_train = tanimoto_matrix([all_mol_info[smi]['ecfp_scaffold'] for smi in all_library_smiles],
                                       [all_mol_info[smi]['ecfp_scaffold'] for smi in train_smiles],
                                       take_mean=True)

    # CATS sim between every screening mol and the training mols
    train_cats = [all_mol_info[smi]['cats'] for smi in train_smiles]
    CATS_S = []
    for smi_i in tqdm(all_library_smiles, desc='\tComputing CATS similarity'):
        cats_i = all_mol_info[smi_i]['cats']
        s_i = cosine_similarity(np.array([cats_i]), train_cats)
        CATS_S.append(np.mean(s_i))
    CATS_S = np.array(CATS_S)

    # MCSF sim between every screening mol and the training mols
    MCSF_S = []
    train_mols = [all_mol_info[smi]['mol'] for smi in train_smiles]
    for smi_i in tqdm(all_library_smiles, desc='\tComputing MCS fraction'):
        mol_i = all_mol_info[smi_i]['mol']
        Si = bulk_mcsf(mol_i, train_mols, symmetric=False)
        MCSF_S.append(np.mean(Si))
    MCSF_S = np.array(MCSF_S)

    df = {'smiles': all_library_smiles,
          'Tanimoto_to_train': T_to_train,
          'Tanimoto_scaffold_to_train': T_scaff_to_train,
          'Cats_cos': CATS_S,
          'MCSF': MCSF_S}

    return pd.DataFrame(df)


if __name__ == '__main__':

    os.chdir(ROOTDIR)

    SPECS_PATH = "data/screening_libraries/specs_cleaned.csv"
    ASINEX_PATH = "data/screening_libraries/asinex_cleaned.csv"
    ENAMINE_HIT_LOCATOR_PATH = "data/screening_libraries/enamine_hit_locator_cleaned.csv"
    output_PATH = 'data/datasets_with_metrics'

    all_datasets = get_all_dataset_names()

    # Load libraries
    library_specs = MoleculeDataset(pd.read_csv(SPECS_PATH)['smiles_cleaned'].tolist(),
                                    descriptor='smiles', randomize_smiles=False)

    library_asinex = MoleculeDataset(pd.read_csv(ASINEX_PATH)['smiles_cleaned'].tolist(),
                                     descriptor='smiles', randomize_smiles=False)

    library_enamine_hit_locator = MoleculeDataset(pd.read_csv(ENAMINE_HIT_LOCATOR_PATH)['smiles_cleaned'].tolist(),
                                                  descriptor='smiles', randomize_smiles=False)

    all_library_smiles = list(set(library_specs.smiles + library_asinex.smiles + library_enamine_hit_locator.smiles))
    all_training_smiles = get_all_train_smiles()

    # precompute all fingerprints and stuff
    all_mol_info = get_all_mol_info(all_library_smiles, all_training_smiles)

    # comptute all distances to the train set
    for i, dataset_name in enumerate(all_datasets):
        print(f'{i}/{len(all_datasets)} - {dataset_name}: ', file=sys.stderr)
        filename = ospj(output_PATH, f"{dataset_name}_library_distances.csv")

        if os.path.exists(filename):
            pass
        else:
            df_dataset = compute_train_library_distance(all_library_smiles, dataset_name, all_mol_info)
            df_dataset.to_csv(filename, index=False)
