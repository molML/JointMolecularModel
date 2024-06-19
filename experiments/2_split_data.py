

import os
import subprocess
from collections import defaultdict, Counter
import pandas as pd
from rdkit import Chem
from scipy.cluster.hierarchy import linkage, dendrogram
from cheminformatics.utils import tanimoto_matrix
from cheminformatics.splitting import map_scaffolds
from cheminformatics.descriptors import mols_to_ecfp


def run_r_script(dataset_name: str, in_path: str, out_path: str, plot_path: str):
    r_script_path = 'experiments/2_cluster_plot.R'
    # Construct the command to run the R script and run it
    command = ['Rscript', r_script_path, dataset_name, in_path, out_path, plot_path]
    result = subprocess.run(command, capture_output=True, text=True)
    # Print errors (if any)
    print("Errors:\n", result.stderr)


if __name__ == '__main__':

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
        out_path = dataset.replace('.csv', '_clustering.csv')
        df_scaffs.to_csv(out_path, index=False)

        # Run an R file that performs hierarchical clustering and visualizes the dendrogram at the same time.
        run_r_script(dataset_name=dataset,
                     in_path=out_path,
                     out_path=out_path,
                     plot_path=f'{dataset}_clustering.pdf')

        # read the file again, now containing clusters
        df_scaffs = pd.read_csv(out_path)

        # Find all SMILES that belong to a cluster
        clustered_smiles = defaultdict(list)
        for smi, clust in zip(df_scaffs['original_smiles'], df_scaffs['clusters']):
            clustered_smiles[clust].extend(smi.split(';'))

        cluster_size_frac = {k: v / len(smiles) for k, v in Counter(df_scaffs['clusters']).items()}

        # Find which cluster is closest to a target value (in percent)
        cluster_size_target = 0.2
        closest_cluster = min(cluster_size_frac, key=lambda k: abs(cluster_size_frac[k] - cluster_size_target))






# Complete Linkage (Maximum Linkage): This method defines the distance between two clusters as the maximum distance
# between any single element in one cluster and any single element in the other cluster. It tends to create more
# compact and spherical clusters, which can be useful when the goal is to find well-separated groups of molecules.

