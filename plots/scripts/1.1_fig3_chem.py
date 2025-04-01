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
from rdkit import Chem
import torch
from cheminformatics.molecular_similarity import tani_sim_to_train, mean_cosine_cats_to_train
from cheminformatics.multiprocessing import tanimoto_matrix, bulk_mcsf
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


def get_MCSF_database(dataset_names: list[str]):
    # MCSF to for every positive test_ood mol to every positive training mol

    database = {}
    for dataset in tqdm(dataset_names):

        df = pd.read_csv(os.path.join('data', 'split', f"{dataset}_split.csv"))

        df_train_hit_smiles = df[(df['split'] == 'train') & df['y'] == 1].smiles.tolist()
        df_test_ood_hit_smiles = df[(df['split'] == 'ood') & df['y'] == 1].smiles.tolist()

        mol_library = {smi: Chem.MolFromSmiles(smi) for smi in df_train_hit_smiles + df_test_ood_hit_smiles}

        MCSF_to_train_hits = {}

        train_mols = [mol_library[smi_j] for smi_j in df_train_hit_smiles]
        for smi_i in tqdm(df_test_ood_hit_smiles):
            Si = bulk_mcsf(mol_library[smi_i], train_mols)
            MCSF_to_train_hits[smi_i] = np.mean(Si)

        database[dataset] = MCSF_to_train_hits

    return database


if __name__ == '__main__':

    # Move to root dir
    os.chdir(ROOTDIR)

    top_k = 50

    df = pd.read_csv('plots/data/df_3_abc.csv')
    dataset_names = set(df['dataset'])

    binning_methods = set(df['ranking_method'])

    # MCSF_hits_database = get_MCSF_database(dataset_names)
    # torch.save(MCSF_hits_database, 'plots/data/MCSF_hit_database.pkl')

    MCSF_hits_database = torch.load('plots/data/MCSF_hit_database.pkl')

    all_results = []
    for dataset in tqdm(dataset_names):
        # train set
        df_train = pd.read_csv(os.path.join('data', 'split', f"{dataset}_split.csv"))
        df_train = df_train[df_train['split'] == 'train']

        for reliability_method in binning_methods:
            try:
                # results dict to store all metrics
                results = {'dataset': dataset, 'ranking_method': reliability_method}

                # subset of selected molecules for screening
                df_subset = df[(df['dataset'] == dataset) & (df['ranking_method'] == reliability_method) & (df['split'] == 'OOD')].copy()

                # Take the top k molecules with the smallest utopia distance
                df_top_k = df_subset.nsmallest(top_k, 'utopia_dist')

                # Create a scatter plot
                # plt.figure(figsize=(8, 6))
                # plt.scatter(df_subset['y_E_'], df_subset['reliability'], color='blue')  # Default color for all points
                # plt.scatter(df_top_k['y_E_'], df_top_k['reliability'], color='red')  # Highlight top 100
                # plt.ylabel(f'{reliability_method}')
                # plt.xlabel('E')
                # plt.title(f'{dataset}')
                # plt.show()

                # full test set
                df_test = df[(df['dataset'] == dataset) & (df['split'] == 'OOD')]
                df_test = df_test[df_test['ranking_method'] == reliability_method]

                # confusion metrics
                P = sum(df_top_k['y'])
                TP = sum((df_top_k['y_hat'] >= 0.5) & (df_top_k['y'] == 1))
                FP = sum((df_top_k['y_hat'] >= 0.5) & (df_top_k['y'] == 0))
                N = len(df_top_k)

                results['N'] = N

                # Hit rate, TP / P
                results['Hit rate'] = TP/P   # correct formula

                # Precision, TP / (TP + FP)
                results['Precision'] = TP / (TP + FP)  # correct formula

                # enrichment. Perform MC sampling by taking 1000 random subsets the same size of the screening subset
                n_mc = 1000
                n_rand_hits = sum([sum(df_test.sample(N)['y']) for _ in range(n_mc)])/n_mc
                results['Enrichment'] = TP/n_rand_hits

                top_k_smiles = df_top_k['smiles'].tolist()
                hit_smiles = df_top_k[(df_top_k['y_hat'] >= 0.5) & (df_top_k['y'] == 1)]['smiles'].tolist()
                train_smiles = df_train['smiles'].tolist()
                train_smiles_actives = df_train[(df_train['y'] == 1)]['smiles'].tolist()

                # ECFP Tanimoto similarity
                results['Tanimoto_topk_to_train'] = df_top_k['Tanimoto_to_train_'].mean()
                results['Tanimoto_hits_to_train_actives'] = tani_sim_to_train(hit_smiles, train_smiles_actives, scaffold=False).mean()

                results['Tanimoto_topk_scaffold_to_train'] = df_top_k['Tanimoto_scaffold_to_train_'].mean()
                results['Tanimoto_hits_scaffold_to_train_actives'] = tani_sim_to_train(hit_smiles, train_smiles_actives, scaffold=True).mean()

                # Cats Cosine similarity
                results['pharmacophore_topk_to_train'] = df_top_k['Cats_cos_'].mean()
                results['pharmacophore_hits_to_train_actives'] = mean_cosine_cats_to_train(hit_smiles, train_smiles_actives).mean()

                # Maximum Common Substructure Fraction (MCSF)
                results['MCSF'] = df_top_k['MCSF_'].mean()
                results['MCSF_hits_to_train_actives'] = np.array([MCSF_hits_database[dataset][smi] for smi in hit_smiles]).mean()

                # diversity (mean Tani) within selected
                results['internal_tanimoto_topk'] = internal_tanimoto(top_k_smiles)
                results['internal_tanimoto_hits'] = internal_tanimoto(hit_smiles)

                topk_scaffolds = mols_to_smiles([get_scaffold(mol) for mol in smiles_to_mols(top_k_smiles)])
                hits_scaffolds = mols_to_smiles([get_scaffold(mol) for mol in smiles_to_mols(hit_smiles)])
                train_scaffolds = mols_to_smiles([get_scaffold(mol) for mol in smiles_to_mols(train_smiles)])
                train_actives_scaffolds = mols_to_smiles([get_scaffold(mol) for mol in smiles_to_mols(train_smiles_actives)])

                # n of unique scaffolds selected
                results['unique_scaffolds_subset'] = len(set(topk_scaffolds))

                # ratio of unique scaffolds selected
                results['unique_scaffolds_subset_ratio'] = len(set(topk_scaffolds))/len(topk_scaffolds)

                # ratio of unique hit scaffolds selected
                results['unique_scaffolds_hits_ratio'] = len(set(hits_scaffolds)) / len(hits_scaffolds)

                # ratio of new scaffolds to train
                results['new_scaffolds_ratio'] = sum([scaf not in train_scaffolds for scaf in topk_scaffolds])/len(topk_scaffolds)

                # ratio of new hit scaffolds
                results['new_hit_scaffolds_ratio'] = sum([scaf not in train_actives_scaffolds for scaf in hits_scaffolds])/len(hits_scaffolds)

                all_results.append(results)

                df_all_results = pd.DataFrame(all_results)
                df_all_results.to_csv(os.path.join('plots', 'data', f'df_3_de.csv'),
                                      index=False)

            except:
                print(f'failed {dataset}, {reliability_method}')
