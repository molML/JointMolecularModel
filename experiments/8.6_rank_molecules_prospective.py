""" Rank molecules for prospective screening using utopia point distance with uncertainty and unfamiliarity
(+ expected value of course, we only want molecules that are predicted as hits).

For each protein target, PIM1 (CHEMBL2147_Ki), CDK1 (CHEMBL308_Ki), and MNK1 (CHEMBL4718_Ki), we select the top
molecules predicted as hits according to:
- Least uncertain + Least unfamiliar
- Least uncertain + Most unfamiliar
- Most uncertain + Least unfamiliar

We skip over all molecules that are too similar to molecules we already selected or too similar to the training data

Derek van Tilborg
Eindhoven University of Technology
April 2025
"""

import os
from os.path import join as ospj
from itertools import chain, batched
from scipy.stats import rankdata
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from cheminformatics.multiprocessing import tanimoto_matrix
from cheminformatics.eval import plot_molecules_acs1996_grid
from cheminformatics.utils import smiles_to_mols
from constants import ROOTDIR


def rank_with_similarity_filter(smiles_list, score_list, similarity_threshold=0.8, stop_at: int = 1000):

    assert len(smiles_list) == len(score_list)

    # Initialize Morgan fingerprint generator
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    # Generate molecules and fingerprints
    fingerprints = {smi: morgan_gen.GetFingerprint(smiles_to_mols(smi)) for smi in smiles_list}

    original_ranks = rankdata(score_list, method='min')
    ranked_smiles = [smi for smi, val in sorted(zip(smiles_list, original_ranks), key=lambda x: x[1])]

    assigned_ranks = []
    selected_fps = []
    current_rank = 1

    for smi in tqdm(ranked_smiles):
        if current_rank <= stop_at:
            fp = fingerprints[smi]
            # Efficient short-circuit similarity check
            if any(DataStructs.TanimotoSimilarity(fp, prev_fp) >= similarity_threshold for prev_fp in selected_fps):
                assigned_ranks.append(None)
            else:
                assigned_ranks.append(current_rank)
                current_rank += 1
                selected_fps.append(fp)
        else:
            assigned_ranks.append(None)


    df = pd.DataFrame({'smiles': smiles_list, 'score': score_list, 'rank': original_ranks})
    df_ = pd.DataFrame({'smiles': ranked_smiles, 'score': score_list, 'assigned_rank': assigned_ranks})

    # combine the distances with the inference results
    df = df.merge(df_, how="left", on=["smiles"])

    return df['assigned_rank'].to_list()


def max_sim_to_dataset(smiles, dataset_name):
    df = pd.read_csv(ospj(f'data/clean/{dataset_name}.csv'))
    dataset_smiles = df.smiles.tolist()

    # Initialize Morgan fingerprint generator
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    # Generate molecules and fingerprints
    dataset_ecfps = [morgan_gen.GetFingerprint(smiles_to_mols(smi)) for smi in dataset_smiles]
    top_n_ecfps = [morgan_gen.GetFingerprint(smiles_to_mols(smi)) for smi in smiles]

    T_to_train = tanimoto_matrix(top_n_ecfps, dataset_ecfps, take_mean=False)

    return np.max(T_to_train, 1)


def column_major_reorder(lst, group_size):
    num_groups = len(lst) // group_size
    grouped = [lst[i * group_size:(i + 1) * group_size] for i in range(num_groups)]
    return list(chain.from_iterable(zip(*grouped)))


def calc_utopia_dist(*params, maximize=None):
    """
    Compute a Euclidean 'utopia distance' in N-dimensional space for any number of parameters.

    Each position i across the parameters will get a single distance value.
    The distance in each dimension is computed based on a [0..1] scale, where we invert
    or not depending on whether we want higher or lower values to be "better."

    :param params: one or more array-like sequences (e.g., lists, NumPy arrays).
                   Each param must have the same length.
    :param maximize: A list or tuple of booleans the same length as 'params', indicating
                     whether each parameter is "higher is better" (True) or "lower is better" (False).
                     If None, default to all True.
    :return: A NumPy array of shape (len_of_params[0],), giving the Euclidean distance
             for each row across all dimensions.

    Example:
        y_E = [10, 8, 12, 15]
        confidence = [0.8, 0.75, 0.5, 0.9]
        param3 = [100, 200, 150, 300]

        # Suppose we want to maximize y_E, maximize confidence, but minimize param3
        dists = calc_utopia_dist(y_E, confidence, param3, maximize=[True, True, False])
    """

    # If user didn't specify a maximize list, default to all True
    num_params = len(params)
    if maximize is None:
        maximize = [True] * num_params

    if len(maximize) != num_params:
        raise ValueError("Length of 'maximize' must match the number of input parameters.")

    # Convert each parameter to a NumPy array
    arrays = [np.asarray(p, dtype=float) for p in params]

    # Check that all parameters have the same length
    length = len(arrays[0])
    if any(len(arr) != length for arr in arrays):
        raise ValueError("All parameters must have the same length.")

    # Normalize each parameter to [0,1], flipping if maximize[i] == True
    normalized_params = []
    for arr, do_max in zip(arrays, maximize):

        # Find min/max
        arr_min, arr_max = arr.min(), arr.max()
        value_range = arr_max - arr_min

        if np.isclose(value_range, 0.0):
            # Edge case: all values are identical.
            # If all values are the same, they effectively contribute 0 to the distance
            # because there's no spread in this dimension.
            norm = np.zeros_like(arr)
        else:
            if do_max:
                # Higher is better -> invert scale
                norm = (arr_max - arr) / value_range
            else:
                # Lower is better -> normal scale
                norm = (arr - arr_min) / value_range

        normalized_params.append(norm)

    # Stack and compute Euclidean distance across all parameters
    # normalized_params is a list of arrays, each shape (length,).
    # We combine them into shape (num_params, length), then do sqrt of sum of squares along axis=0.
    stacked = np.vstack(normalized_params)  # shape = (num_params, length)
    dist = np.sqrt(np.sum(stacked ** 2, axis=0))  # shape = (length,)

    return dist


def get_all_ecfps(smiles: list[str]):

    all_mol_info_path = ospj('data', 'screening_libraries', 'specs_2025', 'ecfps.pt')

    if os.path.exists(all_mol_info_path):
        all_mol_info = torch.load(all_mol_info_path)
    else:
        all_smiles = list(set(smiles))

        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

        all_mol_info = {}
        for smi in tqdm(all_smiles):
            try:
                mol = smiles_to_mols(smi)
                ecfp = mfpgen.GetFingerprint(mol)
                all_mol_info[smi] = {'mol': mol, 'ecfp': ecfp}
            except:
                print(f"failed {smi}")

        torch.save(all_mol_info, all_mol_info_path)

    return all_mol_info


def compute_dataset_library_distance(all_library_smiles, dataset_name, all_mol_info):

    # get the SMILES from the pre-processed file
    data_path = ospj(f'data/clean/{dataset_name}.csv')
    df = pd.read_csv(data_path)
    dataset_smiles = df.smiles.tolist()

    # if there are any failed smiles, remove them
    dataset_smiles = [smi for smi in dataset_smiles if smi in all_mol_info]
    all_library_smiles = [smi for smi in all_library_smiles if smi in all_mol_info]

    # chunk all smiles in batches for efficient multi-threading with manageable memory
    all_library_smiles_batches = [i for i in batched(all_library_smiles, 10000)]

    # tanimoto sim between every screening mol and the training mols
    train_ecfps = [all_mol_info[smi]['ecfp'] for smi in dataset_smiles]
    ECFPs_S_mean = []
    ECFPs_S_max = []
    for batch in tqdm(all_library_smiles_batches, desc='\tComputing Tanimoto similarity', unit_scale=10000):
        T_to_train = tanimoto_matrix([all_mol_info[smi]['ecfp'] for smi in batch],
                                     train_ecfps, take_mean=False)
        ECFPs_S_mean.append(np.mean(T_to_train, 1))
        ECFPs_S_max.append(np.max(T_to_train, 1))

    ECFPs_S_mean = np.concatenate(ECFPs_S_mean)
    ECFPs_S_max = np.concatenate(ECFPs_S_max)

    df = {'smiles': all_library_smiles,
          'dataset': dataset_name,
          'Tanimoto_to_dataset_mean': ECFPs_S_mean,
          'Tanimoto_to_dataset_max': ECFPs_S_max
          }

    return pd.DataFrame(df)



if __name__ == '__main__':

    os.chdir(ROOTDIR)

    # filter out molecules too similar to the train (and to each other during ranking)
    TANIMOTO_CUTOFF = 0.7

    # Select the top 10 ranked molecules
    TOP_N = 10

    data_dir = os.path.join('results', 'prospective')

    datasets = ['CHEMBL4718_Ki', 'CHEMBL308_Ki', 'CHEMBL2147_Ki']
    library_names = ['specs', 'asinex', 'enamine_hit_locator']

    # Load screening results
    screening_results = pd.read_csv(os.path.join(data_dir, 'all_screening_results.csv'))

    # Get all SMILES (datasets + library) and precompute ecfps
    all_dataset_smiles = list(
        set(sum([pd.read_csv(ospj(f'data/clean/{ds}.csv')).smiles.tolist() for ds in datasets], [])))
    all_smiles = screening_results.smiles_cleaned.tolist() + all_dataset_smiles
    all_ecfps = get_all_ecfps(all_smiles)

    # Compute all distances and combine them with the inference results
    dataset_distances = []
    for dataset_name in datasets:
        dataset_distances.append(compute_dataset_library_distance(all_smiles, dataset_name, all_ecfps))
    dataset_distances = pd.concat(dataset_distances)
    screening_results = screening_results.merge(dataset_distances, how="left", on=["smiles", "dataset"])

    # remove molecules that are too similar
    print(f"Removing molecules with a Tanimoto > {TANIMOTO_CUTOFF} ({sum(screening_results['Tanimoto_to_dataset_max'] > TANIMOTO_CUTOFF)} molecules)")
    screening_results = screening_results[screening_results['Tanimoto_to_dataset_max'] < TANIMOTO_CUTOFF]

    ##### calculate utopia distance #####
    ranked_data = []
    for dataset in datasets:
        screening_subset = screening_results[screening_results['dataset'] == dataset].copy()

        # Least uncertain + Least unfamiliar
        screening_subset['Least_uncertain_least_unfamiliar'] = calc_utopia_dist(screening_subset['y_E'],
                                                                                screening_subset['y_unc'],
                                                                                screening_subset['unfamiliarity'],
                                                                                maximize=[True, False, False])
        screening_subset['Least_uncertain_least_unfamiliar_ranked'] = rank_with_similarity_filter(screening_subset['smiles'],
                                                                                                  screening_subset['Least_uncertain_least_unfamiliar'],
                                                                                                  similarity_threshold=TANIMOTO_CUTOFF)

        # Least uncertain + Most unfamiliar
        screening_subset['Least_uncertain_most_unfamiliar'] = calc_utopia_dist(screening_subset['y_E'],
                                                                               screening_subset['y_unc'],
                                                                               screening_subset['unfamiliarity'],
                                                                               maximize=[True, False, True])
        screening_subset['Least_uncertain_most_unfamiliar_ranked'] = rank_with_similarity_filter(screening_subset['smiles'],
                                                                                                 screening_subset['Least_uncertain_most_unfamiliar'],
                                                                                                 similarity_threshold=TANIMOTO_CUTOFF)

        # Most uncertain + Least unfamiliar
        screening_subset['Most_uncertain_least_unfamiliar'] = calc_utopia_dist(screening_subset['y_E'],
                                                                               screening_subset['y_unc'],
                                                                               screening_subset['unfamiliarity'],
                                                                               maximize=[True, True, False])
        screening_subset['Most_uncertain_least_unfamiliar_ranked'] = rank_with_similarity_filter(screening_subset['smiles'],
                                                                                                 screening_subset['Most_uncertain_least_unfamiliar'],
                                                                                                 similarity_threshold=TANIMOTO_CUTOFF)

        # Rank the best molecules
        ranked_cols = [col for col in screening_subset.columns if 'ranked' in col]
        top_n = screening_subset[(screening_subset[ranked_cols] <= TOP_N).any(axis=1)].copy()
        top_n['max_tanimoto_to_dataset'] = max_sim_to_dataset(top_n.smiles, dataset)

        # Get the top n molecules and the legends for a plot
        mols_a = smiles_to_mols(top_n.sort_values('Least_uncertain_least_unfamiliar_ranked')[:TOP_N]['smiles'])
        legends_a = [f"{int(r)}. E: {i: .2f}, Unc: {j: .2f}, Unf: {k: .2f}, Sim: {l: .2f}" for r, i, j, k, l in zip(
            top_n.sort_values('Least_uncertain_least_unfamiliar_ranked')[:TOP_N]['Least_uncertain_least_unfamiliar_ranked'],
            top_n.sort_values('Least_uncertain_least_unfamiliar_ranked')[:TOP_N]['y_E'],
            top_n.sort_values('Least_uncertain_least_unfamiliar_ranked')[:TOP_N]['y_unc'],
            top_n.sort_values('Least_uncertain_least_unfamiliar_ranked')[:TOP_N]['unfamiliarity'],
            top_n.sort_values('Least_uncertain_least_unfamiliar_ranked')[:TOP_N]['max_tanimoto_to_dataset'])]

        mols_b = smiles_to_mols(top_n.sort_values('Least_uncertain_most_unfamiliar_ranked')[:TOP_N]['smiles'])
        legends_b = [f"{int(r)}. E: {i: .2f}, Unc: {j: .2f}, Unf: {k: .2f}, Sim: {l: .2f}" for r, i, j, k, l in zip(
            top_n.sort_values('Least_uncertain_most_unfamiliar_ranked')[:TOP_N]['Least_uncertain_most_unfamiliar_ranked'],
            top_n.sort_values('Least_uncertain_most_unfamiliar_ranked')[:TOP_N]['y_E'],
            top_n.sort_values('Least_uncertain_most_unfamiliar_ranked')[:TOP_N]['y_unc'],
            top_n.sort_values('Least_uncertain_most_unfamiliar_ranked')[:TOP_N]['unfamiliarity'],
            top_n.sort_values('Least_uncertain_most_unfamiliar_ranked')[:TOP_N]['max_tanimoto_to_dataset'])]

        mols_c = smiles_to_mols(top_n.sort_values('Most_uncertain_least_unfamiliar_ranked')[:TOP_N]['smiles'])
        legends_c = [f"{int(r)}. E: {i: .2f}, Unc: {j: .2f}, Unf: {k: .2f}, Sim: {l: .2f}" for r, i, j, k, l in zip(
            top_n.sort_values('Most_uncertain_least_unfamiliar_ranked')[:TOP_N]['Most_uncertain_least_unfamiliar_ranked'],
            top_n.sort_values('Most_uncertain_least_unfamiliar_ranked')[:TOP_N]['y_E'],
            top_n.sort_values('Most_uncertain_least_unfamiliar_ranked')[:TOP_N]['y_unc'],
            top_n.sort_values('Most_uncertain_least_unfamiliar_ranked')[:TOP_N]['unfamiliarity'],
            top_n.sort_values('Most_uncertain_least_unfamiliar_ranked')[:TOP_N]['max_tanimoto_to_dataset'])]

        all_mols = column_major_reorder(mols_a + mols_b + mols_c, TOP_N)  # reorder from [a a a b b b] to [a b a b a b]
        all_legends = column_major_reorder(legends_a + legends_b + legends_c, TOP_N)

        plot_molecules_acs1996_grid(all_mols,
                                    subImgSize=(400, 300),
                                    molsPerRow=3,
                                    legends=all_legends
                                    ).save(ospj(data_dir, f'{dataset}_specs_top_n.png'), dpi=(500, 500))

        screening_subset.to_csv(ospj(data_dir, f'{dataset}_specs_ranked.csv'))
        top_n.to_csv(ospj(data_dir, f'{dataset}_specs_top_n.csv'))
