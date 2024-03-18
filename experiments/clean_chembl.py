
import os
import pandas as pd
from tqdm import tqdm
from dataprep.molecule_processing import clean_mols
from dataprep.utils import smiles_to_mols, mols_to_scaffolds, mols_to_smiles
from dataprep.splitting import scaffold_split, random_split


if __name__ == '__main__':

    # CLEANING ChEMBL ##################################################################################################

    # Read ChEMBL 33
    chembl_smiles = pd.read_table("data/ChEMBL/chembl_33_chemreps.txt").canonical_smiles.tolist()

    # Clean smiles and get rid of duplicates
    chembl_smiles_clean, chembl_smiles_failed = clean_mols(chembl_smiles)
    ''' Cleaned 2,372,125 molecules, failed cleaning 549 molecules:
            reason: 'Strange character': 548, 'Other': 1 ('Cc1ccc2c(c1)-n1-c(=O)/c=c\\c(=O)-n-2-c2cc(C)ccc2-1')
    '''

    len(chembl_smiles_failed['original'])

    chembl_smiles_clean = list(set(chembl_smiles_clean['clean']))
    chembl_smiles_clean = [smi for smi in chembl_smiles_clean if type(smi) is str]
    ''' Out of 2,372,125 SMILES, 2,174,375 were unique '''

    # Save cleaned SMILES strings to a csv file for later use
    pd.DataFrame({'smiles': chembl_smiles_clean}).to_csv("data/ChEMBL/chembl_33_clean.csv")

    chembl_smiles_clean = pd.read_csv("data/ChEMBL/chembl_33_clean.csv").smiles.tolist()

    # CLEANING MoleculeACE #############################################################################################

    moleculeace_datasets = [f'data/moleculeace/{i}' for i in os.listdir('data/moleculeace') if i.startswith('CHEMBL')]
    all_moleculeace_smiles = []

    for filename in moleculeace_datasets:
        df = pd.read_csv(filename)
        ma_smiles_clean, ma_smiles_failed = clean_mols(df.smiles.tolist())
        all_moleculeace_smiles.extend(ma_smiles_clean['clean'])
        df.smiles = ma_smiles_clean['clean']
        # df.to_csv(filename)

    # Get the unqiue scaffolds for all MoleculeACE molecules

    moleculeace_scaffolds = mols_to_scaffolds(smiles_to_mols(all_moleculeace_smiles))

    moleculeace_scaffolds = mols_to_smiles(moleculeace_scaffolds)  # len(moleculeace_scaffolds) -> 48,714
    moleculeace_scaffolds = set(moleculeace_scaffolds)  # len(moleculeace_scaffolds) -> 15,178


    # remove SMILES from the ChEMBL data that occur in MoleculeACE
    chembl_scaffolds = set()
    chembl_not_in_moleculeace = []
    for smi in tqdm(chembl_smiles_clean):
        scaff_smi = mols_to_smiles(mols_to_scaffolds([smiles_to_mols(smi)])[0])
        chembl_scaffolds.add(scaff_smi)
        if scaff_smi not in moleculeace_scaffolds:
            chembl_not_in_moleculeace.append(smi)

    # len(chembl_scaffolds) -> 704,124
    # len(chembl_not_in_moleculeace) -> 1,889,551
    pd.DataFrame({'smiles': chembl_not_in_moleculeace}).to_csv("data/ChEMBL/chembl_33_not_in_moleculeace.csv")
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

    pd.DataFrame({'smiles': train_smiles}).to_csv("data/ChEMBL/chembl_train_smiles.csv")
    pd.DataFrame({'smiles': test_smiles}).to_csv("data/ChEMBL/chembl_test_smiles.csv")
    pd.DataFrame({'smiles': val_smiles}).to_csv("data/ChEMBL/chembl_val_smiles.csv")
