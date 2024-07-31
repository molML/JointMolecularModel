

import os
import numpy as np
from os.path import join as ospj
import subprocess
from collections import Counter
import pandas as pd
from rdkit import Chem
from cheminformatics.utils import tanimoto_matrix
from cheminformatics.splitting import map_scaffolds
from cheminformatics.descriptors import mols_to_ecfp
from constants import ROOTDIR
from scipy.sparse import csgraph
from scipy.linalg import eigh
from kneed import KneeLocator
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import train_test_split


def organize_dataframe(uniques: dict[{str: list}], smiles: list[str]) -> pd.DataFrame:
    """ Puts all scaffolds next to their original smiles in a dataframe. Lists of original SMILES are joined by ';'

    :param uniques: dict with scaffold smiles and the indices to original SMILES strings
    :param smiles: SMILES strings
    :return: dataframe ['scaffolds', 'original_smiles', 'n']
    """
    # fetch the original SMILES that belongs to each unique scaffold (one:many)
    smiles_beloning_to_scaffs, n_mols_with_scaff, unique_scaffold_smiles = [], [], []
    for scaf, idx_list in uniques.items():
        smiles_beloning_to_scaffs.append(';'.join([smiles[i] for i in idx_list]))
        unique_scaffold_smiles.append(scaf)
        n_mols_with_scaff.append(len(idx_list))

    # Put everything in a dataframe
    df = pd.DataFrame({'scaffolds': unique_scaffold_smiles,
                       'original_smiles': smiles_beloning_to_scaffs,
                       'n': n_mols_with_scaff})

    return df


def eigenvalue_cluster_approx(x: np.ndarray) -> int:
    """ We estimate the number of clusters we need for spectral clustering by using the Eigenvalues of the
    Laplacian. The Eigenvalues give you a nice curve from which we determine the elbow with the kneedle algorithm [1].

    [1] Satopaa, V. et al. (2011). Finding a" kneedle" in a haystack: Detecting knee points in system behavior. In
        2011 31st international conference on distributed computing systems workshops (pp. 166-171). IEEE.

    :param x: Similarity/affinity matrix (Tanimoto similarity matrix). Sqaure matrix
    :return: number of clusters
    """

    # Compute the Laplacian matrix. Make sure to compute the symmetrically normalized Laplacian to get a nice L-shape
    laplacian = csgraph.laplacian(x, normed=True)

    # Perform Eigen-decomposition to get the Eigenvalues and Eigenvectors
    eigenvalues, eigenvectors = eigh(laplacian)

    # Estimate the 'elbow'/'knee' of the curve using the kneedle algorithm.
    kn = KneeLocator(range(len(eigenvalues)), eigenvalues,
                     curve='concave', direction='increasing',
                     interp_method='interp1d', )

    n_clusters = kn.knee

    # df_eig = pd.DataFrame({'index': range(len(eigenvalues)), 'eigenvalues': eigenvalues, 'knee': kn.knee})
    # df_eig.to_csv('results/Eigenvalues_ChEMBL4792_Ki.csv', index=False)

    # import matplotlib.pyplot as plt
    # plt.plot(eigenvalues, marker='o')
    # plt.xlabel('Index')
    # plt.ylabel('Eigenvalue')
    # plt.vlines(n_clusters, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    # plt.show()

    return n_clusters


def cluster_similarity(X: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """ Find the pairwise distance of every cluster.

    :param X: Tanimoto similarity :math:`(N, N)`
    :param clusters: cluster membership vector :math:`(N)`, where every item is cluster membership :math:`c`
    :return: similarity matrix :math:`(c, c)`
    """

    n_clusters = len(set(clusters))

    # which molecules belong to which cluster?
    clust_molidx = {c: np.where(clusters == c)[0] for c in set(clusters)}

    # empty matrix (n_clust x n_clust)
    clust_sims = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(i, n_clusters):
            # get the indices of the scaffolds in this cluster
            row_idx, col_idx = clust_molidx[i], clust_molidx[j]

            # Find the mean inter-cluster similarity
            clust_sim_matrix = X[row_idx][:, col_idx]
            clust_sims[i, j] = np.mean(clust_sim_matrix)

    # Mirror out the lower triangle of the matrix
    clust_sims = clust_sims + clust_sims.T - np.diag(np.diag(clust_sims))

    return clust_sims


def mean_cluster_sim_to_all_clusters(x: np.ndarray) -> np.ndarray:
    """ calculate the mean similarity of each cluster to all other clusters. Masks the diagonal so
    self-similarity is not taken into account """

    n_clusters = len(x)

    mean_clust_sim = []
    for i in range(n_clusters):
        mask = np.array([j for j in range(n_clusters) if j is not i])
        mean_clust_sim.append(np.mean(x[i][mask]))

    return mean_clust_sim


def group_and_sort(clusters, similarity_matrix, n_smiles_for_each_scaffold):
    # Compute the pairwise similarity between whole clusters and the mean similarity of each cluster to all others
    clust_sims = cluster_similarity(similarity_matrix, clusters)
    mean_clust_sims = mean_cluster_sim_to_all_clusters(clust_sims)

    # Get the original size of each cluster
    cluster_size = [sum(np.array(n_smiles_for_each_scaffold)[np.argwhere(clusters == c).flatten()]) for c in
                    set(clusters)]

    # put everything together
    df = pd.DataFrame({'cluster': list(range(len(set(clusters)))),
                       'size (scaffolds)': [i[1] for i in sorted(Counter(clusters).items())],
                       'size (full)': cluster_size,
                       'mean sim': mean_clust_sims})

    df.sort_values(by=['mean sim'], inplace=True)

    return df


def select_ood_clusters(df, size_cutoff):
    selected_clusters = []
    setsize = 0
    for clust, clust_size in zip(df['cluster'], df['size (full)']):

        if (setsize + clust_size) < size_cutoff:
            selected_clusters.append(clust)
            setsize += clust_size

    return selected_clusters


def split_data(clusters: list[int], ood_clusters: list[int]) -> list[int]:
    """
    :param clusters: list of clusters
    :param ood_clusters: list of clusters that should be in the ood

    :return: list of split membership
    """

    ood_idx = [i for i, c in enumerate(clusters) if c in ood_clusters]
    non_ood_idx = [i for i, c in enumerate(clusters) if c not in ood_clusters]

    # split the rest of the data randomly, using a test set that is the same size as the OOD set
    train_idx, test_idx = train_test_split(non_ood_idx, test_size=len(ood_idx), shuffle=True)

    split = np.array(['train'] * len(clusters))
    split[test_idx] = 'test'
    split[ood_idx] = 'ood'

    return split


def split_chembl(df: pd.DataFrame, random_state: int = 1, test_frac: float = 0.1, val_frac: float = 0.1) -> pd.DataFrame:
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
    split_col = (['val'] * int(len(df) * 0.1)) + (['test'] * int(len(df) * 0.1))
    split_col = (['train'] * (len(df) - len(split_col))) + split_col
    df['split'] = split_col

    return df


def split_finetuning_data(df: pd.DataFrame, ood_fraction: float = 0.25) -> pd.DataFrame:
    """ Split finetuning datasets into an ood, test, and train set. OOD molecules are determined using spectral
    clustering

    :param df: dataframe with `smiles`, `y`
    :param ood_fraction: fraction of data that goes to the OOD split. The test set gets the same size as the OOD split,
    and the remainder will become training data
    :return: dataframe with `smiles`, `y`, `cluster`, `split`
    """
    # get the SMILES strings
    smiles = df.smiles.tolist()

    # convert to scaffolds
    scaffold_smiles, uniques = map_scaffolds(smiles, scaffold_type='cyclic_skeleton')

    # Put every original smiles to each unique scaffold
    df_scaffs = organize_dataframe(uniques, smiles)

    # Compute a distance matrix over the scaffolds
    scaffold_mols = [Chem.MolFromSmiles(smi) for smi in df_scaffs['scaffolds']]
    ecfps = mols_to_ecfp(scaffold_mols, radius=2, nbits=2048)
    S = tanimoto_matrix(ecfps, dtype=float)

    # Estimate the number of clusters using the Eigenvalues of the Laplacian
    n_clusters = eigenvalue_cluster_approx(S)

    # Perform spectral clustering
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans')
    clusters = spectral.fit_predict(S)
    df_scaffs['cluster'] = clusters

    # put all clusters in a nice dataframe sorted by their mean similarity to all other clusters.
    df_clusters = group_and_sort(clusters, similarity_matrix=S, n_smiles_for_each_scaffold=df_scaffs['n'])

    # Select clusters by taking the clusters in  successively
    ood_clusters = select_ood_clusters(df_clusters, len(smiles) * ood_fraction)

    # map the clusters back to the original SMILES
    clusters_per_original = ['x'] * len(smiles)
    for originals, c in zip(df_scaffs['original_smiles'], clusters):
        for smi in originals.split(';'):
            clusters_per_original[smiles.index(smi)] = c
    df['cluster'] = clusters_per_original

    # Determine the splits
    split = split_data(clusters_per_original, ood_clusters)
    df['split'] = split

    return df


if __name__ == '__main__':

    OOD_SET_FRACTION = 0.25
    CHEMBL_TEST_FRACTION = 0.10
    CHEMBL_VAL_FRACTION = 0.10

    IN_DIR_PATH = 'data/clean'
    OUT_DIR_PATH = 'data/split'
    os.chdir(ROOTDIR)

    # split ChEMBL
    chembl = pd.read_csv(ospj(IN_DIR_PATH, 'ChEMBL_33_filtered.csv'))
    chembl = split_chembl(chembl, test_frac=CHEMBL_TEST_FRACTION, val_frac=CHEMBL_VAL_FRACTION, random_state=1)
    chembl.to_csv(ospj(OUT_DIR_PATH, 'ChEMBL_33_split.csv'), index=False)

    # Split finetuning datasets
    datasets = [i for i in os.listdir(IN_DIR_PATH) if not i.startswith('ChEMBL_33')]
    for dataset in datasets:
        finetuning_df = pd.read_csv(ospj(IN_DIR_PATH, dataset))
        finetuning_df = split_finetuning_data(finetuning_df, ood_fraction=OOD_SET_FRACTION)
        finetuning_df.to_csv(ospj(OUT_DIR_PATH, dataset.replace('.csv', '_split.csv')))
