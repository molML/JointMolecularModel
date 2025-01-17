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


if __name__ == '__main__':

    # Move to root dir
    os.chdir(ROOTDIR)

    df = pd.read_csv('plots/data/df_3abc.csv')

    # bin 10 has the 'most confident' molecules, i.e. least uncertain and least unfamiliar

    df_confident = df[df['bin'] == 10]
    df_pos_pred = df_confident[df_confident['y_hat'] == 1.0]


    all_results = []
    for dataset in tqdm(set(df_pos_pred['dataset'])):

        # train set
        df_train = pd.read_csv(os.path.join('data', 'split', f"{dataset}_split.csv"))
        df_train = df_train[df_train['split'] == 'train']

        for reliability_method in set(df['reliability_method']):

            try:
                # results dict to store all metrics
                results = {'dataset': dataset, 'reliability_method': reliability_method}

                # subset of selected molecules for screening
                df_subset = df_pos_pred[df_pos_pred['dataset'] == dataset]
                df_subset = df_subset[df_subset['reliability_method'] == reliability_method]

                # full test + ood set
                df_test_ood = df[df['dataset'] == dataset]
                df_test_ood = df_test_ood[df_test_ood['reliability_method'] == reliability_method]

                P = sum(df_subset['y_hat'])
                TP = sum(df_subset['y'])
                N = len(df_subset)
                FP = P - sum(df_subset['y'])

                results['N'] = N

                # Hit rate, TP / P
                results['Hit rate'] = TP/P

                # Precision, TP / (TP + FP)
                results['Precision'] = TP / (TP + FP)

                # enrichment. Perform MC sampling by taking 1000 random subsets the same size of the screening subset
                n_mc = 1000
                n_rand_hits = sum([sum(df_test_ood.sample(N)['y']) for _ in range(n_mc)])/n_mc
                results['Enrichment'] = TP/n_rand_hits

                subset_smiles = df_subset['smiles'].tolist()
                train_smiles = df_train['smiles'].tolist()
                train_smiles_hits = df_train[(df_train['y'] == 1)]['smiles'].tolist()

                # ECFP Tanimoto similarity
                results['Tanimoto_to_train'] = tani_sim_to_train(subset_smiles, train_smiles, scaffold=False).mean()
                results['Tanimoto_to_train_hits'] = tani_sim_to_train(subset_smiles, train_smiles_hits, scaffold=False).mean()

                results['Tanimoto_scaffold_to_train'] = tani_sim_to_train(subset_smiles, train_smiles, scaffold=True).mean()
                results['Tanimoto_scaffold_to_train_hits'] = tani_sim_to_train(subset_smiles, train_smiles_hits, scaffold=True).mean()

                # Cats Cosine similarity
                results['Cats_cos'] = mean_cosine_cats_to_train(subset_smiles, train_smiles).mean()
                results['Cats_cos_hits'] = mean_cosine_cats_to_train(subset_smiles, train_smiles_hits).mean()

                # Maximum Common Substructure Fraction (MCSF)
                results['MCSF'] = mcsf_to_train(subset_smiles, train_smiles, scaffold=False).mean()
                results['MCSF_hits'] = mcsf_to_train(subset_smiles, train_smiles_hits, scaffold=False).mean()

                # diversity (mean Tani) within selected
                results['internal_tanimoto_subset'] = internal_tanimoto(subset_smiles)
                results['internal_tanimoto_train'] = internal_tanimoto(train_smiles)
                results['internal_tanimoto_train_hit'] = internal_tanimoto(train_smiles_hits)

                subset_scaffolds = mols_to_smiles([get_scaffold(mol) for mol in smiles_to_mols(subset_smiles)])
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
                results['new_hit_scaffolds_ratio'] = sum([scaf not in train_scaffolds for scaf in train_hits_scaffolds])/len(subset_scaffolds)

                all_results.append(results)
            except:
                print(f'failed {dataset}, {reliability_method}')

        df_all_results = pd.DataFrame(all_results)
        df_all_results.to_csv(os.path.join('results', 'processed', 'screening_mols_properties.csv'), index=False)
