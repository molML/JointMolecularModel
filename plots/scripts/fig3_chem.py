"""
Analyze what kind of chemistry can be found in the top uncertainty/unfamiliarity bins

Derek van Tilborg
Eindhoven University of Technology
January 2024
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from cheminformatics.molecular_similarity import tani_sim_to_train, mean_cosine_cats_to_train, mcsf_to_train
from cheminformatics.multiprocessing import tanimoto_matrix
from cheminformatics.descriptors import mols_to_ecfp
from cheminformatics.utils import smiles_to_mols, get_scaffold, mols_to_smiles
from constants import ROOTDIR


def internal_tanimoto(smiles: list[str], radius: int = 2, nbits: int = 2048) -> float:
    """ Calculate the internal diversity """

    ecfps = mols_to_ecfp(smiles_to_mols(smiles), radius=radius, nbits=nbits)
    T = tanimoto_matrix(ecfps, ecfps)

    # Create a mask for the diagonal
    mask = ~np.eye(T.shape[0], dtype=bool)

    # Calculate the mean of the non-diagonal elements
    mean_non_diagonal = np.mean(T[mask])

    return mean_non_diagonal


def calc_utopia_dist(y_E, uncertainty) -> np.ndarray:

    X = np.array((np.array(y_E), np.array(uncertainty))).T

    d_utopia = []
    E_max, E_min = max(X[:, 0]), min(X[:, 0])
    unc_max, unc_min = max(X[:, 1]), min(X[:, 1])
    for i in X:
        E_i = i[0]
        unc_i = i[1]

        dist = ((E_max - E_i) / (E_max - E_min)) ** 2 + ((unc_i - unc_min) / (unc_max - unc_min)) ** 2
        d_utopia.append(dist)

    dist_ranking = np.array(d_utopia)

    return dist_ranking


if __name__ == '__main__':

    # Move to root dir
    os.chdir(ROOTDIR)

    top_k = 100
    binning_methods = ['Embedding dist', 'Unfamiliarity', 'Uncertainty', 'Substructure sim', 'Expected value']

    df = pd.read_csv('plots/data/df_3abc.csv')

    # Make sure that all metrics are a lower = better metric.
    # Conveniently, both Tanimoto similarity and expected values range between 0-1
    df.loc[df['reliability_method'] == 'Substructure sim', 'reliability'] = 1 - df.loc[df['reliability_method'] == 'Substructure sim', 'reliability']
    df.loc[df['reliability_method'] == 'Expected value', 'reliability'] = 1 - df.loc[df['reliability_method'] == 'Expected value', 'reliability']

    all_results = []
    for dataset in tqdm(set(df['dataset'])):
        # break
        # train set
        df_train = pd.read_csv(os.path.join('data', 'split', f"{dataset}_split.csv"))
        df_train = df_train[df_train['split'] == 'train']

        for reliability_method in binning_methods:
            # break
            try:
                # results dict to store all metrics
                results = {'dataset': dataset, 'reliability_method': reliability_method}

                # subset of selected molecules for screening
                df_subset = df[(df['dataset'] == dataset) & (df['reliability_method'] == reliability_method)].copy()

                # Rank molecules on their distance to the utopia point (max E[p(y|x)] and min H[p(y|x)])
                utopia_dist = calc_utopia_dist(df_subset['y_E_'], df_subset['reliability'])
                df_subset['ranking'] = utopia_dist

                # Take the top k molecules with the smallest utopia distance
                df_top_k = df_subset.nsmallest(top_k, 'ranking')

                # Create a scatter plot
                plt.figure(figsize=(8, 6))
                plt.scatter(df_subset['y_E_'], df_subset['reliability'], color='blue')  # Default color for all points
                plt.scatter(df_top_k['y_E_'], df_top_k['reliability'], color='red')  # Highlight top 100
                plt.ylabel(f'{reliability_method}')
                plt.xlabel('E')
                plt.title(f'{dataset}')
                plt.show()

                # full test + ood set
                df_test_ood = df[df['dataset'] == dataset]
                df_test_ood = df_test_ood[df_test_ood['reliability_method'] == reliability_method]

                P = sum(df_top_k['y_hat'])
                TP = sum(df_top_k['y'])
                N = len(df_top_k)
                FP = P - sum(df_top_k['y'])

                results['N'] = N

                # Hit rate, TP / P
                results['Hit rate'] = TP/P

                # Precision, TP / (TP + FP)
                results['Precision'] = TP / (TP + FP)

                # enrichment. Perform MC sampling by taking 1000 random subsets the same size of the screening subset
                n_mc = 1000
                n_rand_hits = sum([sum(df_test_ood.sample(N)['y']) for _ in range(n_mc)])/n_mc
                results['Enrichment'] = TP/n_rand_hits

                top_k_smiles = df_top_k['smiles'].tolist()
                train_smiles = df_train['smiles'].tolist()
                train_smiles_hits = df_train[(df_train['y'] == 1)]['smiles'].tolist()

                # ECFP Tanimoto similarity
                results['Tanimoto_to_train'] = df_top_k['Tanimoto_to_train_'].mean()
                results['Tanimoto_to_train_hits'] = tani_sim_to_train(top_k_smiles, train_smiles_hits, scaffold=False).mean()

                results['Tanimoto_scaffold_to_train'] = df_top_k['Tanimoto_scaffold_to_train_'].mean()
                results['Tanimoto_scaffold_to_train_hits'] = tani_sim_to_train(top_k_smiles, train_smiles_hits, scaffold=True).mean()

                # Cats Cosine similarity
                results['Cats_cos'] = df_top_k['Cats_cos_'].mean()
                results['Cats_cos_hits'] = mean_cosine_cats_to_train(top_k_smiles, train_smiles_hits).mean()

                # Maximum Common Substructure Fraction (MCSF)
                results['MCSF'] = df_top_k['MCSF_'].mean()
                # results['MCSF_hits'] = mcsf_to_train(top_k_smiles, train_smiles_hits, scaffold=False).mean()

                # diversity (mean Tani) within selected
                results['internal_tanimoto_subset'] = internal_tanimoto(top_k_smiles)
                results['internal_tanimoto_train'] = internal_tanimoto(train_smiles)
                results['internal_tanimoto_train_hit'] = internal_tanimoto(train_smiles_hits)

                subset_scaffolds = mols_to_smiles([get_scaffold(mol) for mol in smiles_to_mols(top_k_smiles)])
                train_scaffolds = mols_to_smiles([get_scaffold(mol) for mol in smiles_to_mols(train_smiles)])
                train_hits_scaffolds = mols_to_smiles([get_scaffold(mol) for mol in smiles_to_mols(train_smiles_hits)])

                # n new pharmaco/chemical features abscent in the training data
                # ?

                # n of unique scaffolds selected
                results['unique_scaffolds_subset'] = len(set(subset_scaffolds))

                # ratio of unique scaffolds selected
                results['unique_scaffolds_subset_ratio'] = len(set(subset_scaffolds))/len(subset_scaffolds)

                # ratio of new scaffolds to train
                results['new_scaffolds_ratio'] = sum([scaf not in train_scaffolds for scaf in subset_scaffolds])/len(subset_scaffolds)

                # ratio of new hit scaffolds
                results['new_hit_scaffolds_ratio'] = sum([scaf not in train_hits_scaffolds for scaf in subset_scaffolds])/len(subset_scaffolds)

                all_results.append(results)

                df_all_results = pd.DataFrame(all_results)
                df_all_results.to_csv(os.path.join('results', 'screening_mols_properties.csv'),
                                      index=False)

            except:
                print(f'failed {dataset}, {reliability_method}, TP:{TP}')
