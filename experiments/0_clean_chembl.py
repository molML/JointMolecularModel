
import os
from collections import Counter
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from rdkit.DataStructs import BulkTanimotoSimilarity
from cheminformatics.cleaning import clean_mols
from cheminformatics.utils import smiles_to_mols, mols_to_scaffolds, mols_to_smiles
from cheminformatics.splitting import scaffold_split, random_split
from cheminformatics.descriptors import mols_to_ecfp
from cheminformatics.complexity import smile_complexity


def chembl_scaffold_sim(moleculeace_scaffolds: list[str], chembl_smiles: list[str]):
    moleculeace_scaffolds_fps = mols_to_ecfp(smiles_to_mols(moleculeace_scaffolds))

    chembl_scaffolds = set()
    max_sim_to_moleculeace = []
    mean_sim_to_moleculace = []

    for smi in tqdm(chembl_smiles):
        chembl_scaff = mols_to_scaffolds([smiles_to_mols(smi)])[0]
        chembl_scaff_fp = mols_to_ecfp([chembl_scaff])[0]

        M = BulkTanimotoSimilarity(chembl_scaff_fp, moleculeace_scaffolds_fps)

        max_sim_to_moleculeace.append(np.max(M))
        mean_sim_to_moleculace.append(np.mean(M))
        chembl_scaffolds.add(mols_to_smiles(chembl_scaff))

    return chembl_scaffolds, max_sim_to_moleculeace, mean_sim_to_moleculace


if __name__ == '__main__':

    # CLEANING ChEMBL ##################################################################################################

    # Read ChEMBL 33
    chembl_smiles = pd.read_table("data/ChEMBL/chembl_33_chemreps.txt").canonical_smiles.tolist()

    print('started with', len(chembl_smiles))  # 2,372,674
    # Clean smiles and get rid of duplicates
    chembl_smiles_clean, chembl_smiles_failed = clean_mols(chembl_smiles)
    ''' Cleaned 2,139,472 molecules, failed cleaning 233,202 molecules:
            reason: 'Does not fit vocab': 231,632, 'Isotope': 1,516, None: 53, 'Other': 1
    '''

    print('failed', len(chembl_smiles_failed['reason']), Counter(chembl_smiles_failed['reason']))
    print('clean smiles', len(chembl_smiles_clean['clean']))

    chembl_smiles_clean = list(set(chembl_smiles_clean['clean']))
    chembl_smiles_clean = [smi for smi in chembl_smiles_clean if type(smi) is str and smi != '']
    '''1,974,867 were unique '''
    print('uniques', len(chembl_smiles_clean))

    # Save cleaned SMILES strings to a csv file for later use
    pd.DataFrame({'smiles': chembl_smiles_clean}).to_csv("data/ChEMBL/chembl_33_clean.csv", index=False)

    # chembl_smiles_clean = pd.read_csv("data/ChEMBL/chembl_33_clean.csv").smiles.tolist()

    # CLEANING MoleculeACE #############################################################################################

    moleculeace_datasets = [f'data/moleculeace_original/{i}' for i in os.listdir('data/moleculeace_original') if i.startswith('CHEMBL')]
    all_moleculeace_smiles = []

    for filename in moleculeace_datasets:
        df = pd.read_csv(filename)
        smiles = df.smiles.tolist()
        ma_smiles_clean, ma_smiles_failed = clean_mols(smiles)
        all_moleculeace_smiles.extend(ma_smiles_clean['clean'])

        df_passed = df.iloc[[smiles.index(smi) for smi in ma_smiles_clean['original']]]
        df_passed.smiles = ma_smiles_clean['clean']

        df_passed.to_csv(filename.replace('_original', ''), index=False)


    # Get the unqiue scaffolds for all MoleculeACE molecules
    moleculeace_scaffolds = mols_to_scaffolds(smiles_to_mols(all_moleculeace_smiles))

    moleculeace_scaffolds = mols_to_smiles(moleculeace_scaffolds)  # len(moleculeace_scaffolds) -> 48,008
    moleculeace_scaffolds = set(moleculeace_scaffolds)  # len(moleculeace_scaffolds) -> 14,879

    chembl_scaffolds, max_sim_to_moleculeace, mean_sim_to_moleculace = chembl_scaffold_sim(moleculeace_scaffolds,
                                                                                           chembl_smiles_clean)

    # remove molecules with a max scaffold Tanimoto similarity to MoleculeACE scaffolds > 0.6 from ChEMBL
    CUTOFF = 0.6

    allowed_mol_idx = np.argwhere(np.array(max_sim_to_moleculeace) < CUTOFF).flatten()
    chembl_not_in_moleculeace = [chembl_smiles_clean[i] for i in allowed_mol_idx]

    # len(chembl_not_in_moleculeace) -> 1,608,446
    pd.DataFrame({'smiles': chembl_not_in_moleculeace}).to_csv("data/ChEMBL/chembl_33_not_in_moleculeace.csv", index=False)
    # chembl_not_in_moleculeace = pd.read_csv("data/ChEMBL/chembl_33_not_in_moleculeace.csv").smiles.tolist()

    # Splitting data ###################################################################################################

    # Split ChEMBL in a train and test split using a scaffold split
    train_idx, test_idx = scaffold_split(chembl_not_in_moleculeace, ratio=0.1)

    train_smiles = [chembl_not_in_moleculeace[i] for i in train_idx]
    test_smiles = [chembl_not_in_moleculeace[i] for i in test_idx]

    # Split the train split further into a train and val split, but now using random splitting
    train_idx, val_idx = random_split(train_smiles, ratio=0.01)
    val_smiles = [train_smiles[i] for i in val_idx]
    train_smiles = [train_smiles[i] for i in train_idx]

    train_smiles = [i for i in train_smiles if type(i) is str]
    test_smiles = [i for i in test_smiles if type(i) is str]
    val_smiles = [i for i in val_smiles if type(i) is str]

    pd.DataFrame({'smiles': train_smiles}).to_csv("data/ChEMBL/chembl_train_smiles.csv", index=False)
    pd.DataFrame({'smiles': test_smiles}).to_csv("data/ChEMBL/chembl_test_smiles.csv", index=False)
    pd.DataFrame({'smiles': val_smiles}).to_csv("data/ChEMBL/chembl_val_smiles.csv", index=False)

    # Compute complexity for curriculum learning
    complexity_dict = {smi: smile_complexity(smi) for smi in tqdm(train_smiles)}
    torch.save(complexity_dict, "data/ChEMBL/chembl_train_smiles_complexity.pt")
