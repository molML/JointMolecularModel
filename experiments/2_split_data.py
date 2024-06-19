

import os
import pandas as pd
from rdkit import Chem
from scipy.cluster.hierarchy import linkage, dendrogram
from cheminformatics.utils import tanimoto_matrix
from cheminformatics.splitting import map_scaffolds
from cheminformatics.descriptors import mols_to_ecfp
import matplotlib.pyplot as plt


datasets = [i for i in os.listdir('data/clean') if i != 'ChEMBL_33.csv']
for dataset in datasets:

    # read data
    df = pd.read_csv(f'data/clean/{dataset}')

    # convert to scaffolds
    smiles = df.smiles.tolist()
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    scaffolds, uniques = map_scaffolds(smiles, scaffold_type='bemis_murcko')

    # fetch the original SMILES that belongs to each unique scaffold (one:many)
    smiles_beloning_to_scaffs, n_mols_with_scaff = [], []
    for scaf, idx_list in uniques.items():
        smiles_beloning_to_scaffs.append(';'.join([smiles[i] for i in idx_list]))
        n_mols_with_scaff.append(len(idx_list))

    # Put everything in a dataframe
    df_scaffs = pd.DataFrame({'scaffolds': list(uniques.keys()), 'original_smiles': smiles_beloning_to_scaffs, 'n': n_mols_with_scaff})

    # Compute a distance matrix over the scaffolds
    scaffold_mols = [Chem.MolFromSmiles(smi) for smi in df_scaffs['scaffolds']]
    ecfps = mols_to_ecfp(scaffold_mols, radius=2, nbits=2048)
    S = tanimoto_matrix(ecfps)

    # save file
    df_scaffs = pd.concat([pd.DataFrame(S), df_scaffs], axis=1)
    df_scaffs.to_csv(f'CHEMBL2047_EC50_mol_dist.csv', index=False)

    dataset.replace('.csv', '_sim_matrix.csv')

    # Run an R file that performs hierarchical clustering and visualizes the dendrogram at the same time.
    ...

    # import results and determine the data split based on the clustering





# Complete Linkage (Maximum Linkage): This method defines the distance between two clusters as the maximum distance
# between any single element in one cluster and any single element in the other cluster. It tends to create more
# compact and spherical clusters, which can be useful when the goal is to find well-separated groups of molecules.

