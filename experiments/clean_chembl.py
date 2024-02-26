import numpy as np
import pandas as pd
from molbotomy import SpringCleaning, tools, check_splits, scaffold_split, random_split

if __name__ == '__main__':

    # Read ChEMBL 33
    smiles = pd.read_table("data/chembl_33_chemreps.txt").canonical_smiles.tolist()

    # Clean ChEMBL SMILES strings
    cleaner = SpringCleaning(canonicalize=True, flatten_stereochem=False, neutralize=True, check_for_uncommon_atoms=True,
                             desalt=True, remove_solvent=True, unrepeat=True, sanitize=True)
    smiles = cleaner.clean(smiles)
    smiles = list(set(smiles))

    pd.DataFrame({'smiles': smiles}).to_csv("data/chembl_33_clean_smiles.csv")

    # Parsed 2372674 molecules of which 2361127 successfully.
    # Failed to clean 11547 molecules: {'unfamiliar token': 8922, 'fragmented SMILES': 2624, 'unknown': 1}
    # 2251537 after removing duplicates

    # Split ChEMBL in a train and test split using a scaffold split
    mols = tools.smiles_to_mols(smiles)
    train_idx, test_idx = scaffold_split(mols, ratio=0.1)
    del mols
    train_smiles = [smiles[i] for i in train_idx]
    test_smiles = [smiles[i] for i in test_idx]

    # Split the train split further into a train and val split, but now using random splitting
    train_idx, val_idx = random_split(train_smiles, ratio=0.01)
    val_smiles = [train_smiles[i] for i in val_idx]
    train_smiles = [train_smiles[i] for i in train_idx]

    train_smiles = [i for i in train_smiles if type(i) is str]
    test_smiles = [i for i in test_smiles if type(i) is str]
    val_smiles = [i for i in val_smiles if type(i) is str]

    pd.DataFrame({'smiles': train_smiles}).to_csv("data/train_smiles.csv")
    pd.DataFrame({'smiles': test_smiles}).to_csv("data/test_smiles.csv")
    pd.DataFrame({'smiles': val_smiles}).to_csv("data/val_smiles.csv")

    check_splits(test_smiles, val_smiles)
    # Data leakage:
    # 	Found 0 intersecting SMILES between the train and test set.
    # 	Found 0 intersecting Bemis-Murcko scaffolds between the train and test set.
    # 	Found 22 intersecting stereoisomers between the train and test set.
    # Duplicates:
    # 	Found 0 duplicate SMILES in the train set.
    # 	Found 0 duplicate SMILES in the test set.
    # Stereoisomers:
    # 	Found 5394 Stereoisomer SMILES in the train set.
    # 	Found 17 Stereoisomer SMILES in the test set.

    check_splits(train_smiles, test_smiles)


    # train_smiles = pd.read_csv("data/train_smiles.csv").smiles.tolist()
    # test_smiles = pd.read_csv("data/test_smiles.csv").smiles.tolist()
    # val_smiles = pd.read_csv("data/val_smiles.csv").smiles.tolist()
    #
    # from molbotomy.descriptors import mols_to_ecfp, mols_to_maccs
    # from rdkit import Chem
    # import numpy as np
    # import math

    #
    # type(train_smiles[1324755]) is str
    #
    #
    # for i, smi in enumerate(train_smiles):
    #     try:
    #         m = Chem.MolFromSmiles(smi)
    #         x = mols_to_maccs([m], to_array=True)
    #     except:
    #         print(i, smi)
    #
    #
    # mols = tools.smiles_to_mols(train_smiles)
    # x = mols_to_ecfp(mols, to_array=True)

