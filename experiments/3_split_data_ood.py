"""
1. Cluster MoleculeACE datasets into two clusters with k-medoids from a Tanimoto matrix
2. Take the smallest cluster as the OOD test split
3. Write a column with the split and the distance to the medoid of the train set (for each molecule)

Derek van Tilborg
Eindhoven University of Technology
Feb 2024
"""

import os
from collections import Counter
from tqdm.auto import tqdm
import pandas as pd
import kmedoids
from cheminformatics.utils import smiles_to_mols, tanimoto_matrix
from cheminformatics.descriptors import mols_to_ecfp


SEED = 42

if __name__ == '__main__':

    datasets = [f'data/moleculeace/{i}' for i in os.listdir('data/moleculeace') if i.startswith('CHEMBL')]
    for dataset_path in tqdm(datasets):

        # read dataset
        df = pd.read_csv(dataset_path)
        smiles = df.smiles.tolist()

        # Compute pairwise Tanimoto similarity and cluster using the k-medoids algorithm
        M = tanimoto_matrix(mols_to_ecfp(smiles_to_mols(smiles)), progressbar=False).astype(float)
        clustering = kmedoids.fasterpam(1-M, 2, random_state=SEED)

        # Find the train cluster (biggest cluster) and test (OOD) cluster
        train_cluster = max(Counter(clustering.labels), key=Counter(clustering.labels).get)
        test_cluster = min(Counter(clustering.labels), key=Counter(clustering.labels).get)
        train_medoid = clustering.medoids[train_cluster]

        # Write train/test split and distance to the train medoid to the csv
        df['ood_split'] = ['train' if i == train_cluster else 'test' for i in clustering.labels]
        df['sim_to_train_medoid'] = M[..., train_medoid]
        df.to_csv(dataset_path)
