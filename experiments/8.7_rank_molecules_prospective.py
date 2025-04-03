""" Rank molecules for prospective screening using utopia point distance with uncertainty and unfamiliarity
(+ expected value of course, we only want molecules that are predicted as hits).

For each protein target, PIM1 (CHEMBL2147_Ki), CDK1 (CHEMBL308_Ki), and MNK1 (CHEMBL4718_Ki), we select the top
molecules predicted as hits according to:
- Least uncertain + Least unfamiliar
- Least uncertain + Most unfamiliar
- Most uncertain + Least unfamiliar

Derek van Tilborg
Eindhoven University of Technology
April 2025
"""

import os
from os.path import join as ospj
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from constants import ROOTDIR
from cheminformatics.utils import smiles_to_mols
from itertools import chain
from cheminformatics.eval import plot_molecules_acs1996_grid


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


if __name__ == '__main__':

    os.chdir(ROOTDIR)

    TANIMOTO_CUTOFF = 0.7

    data_dir = os.path.join('results', 'prospective')

    datasets = ['CHEMBL4718_Ki', 'CHEMBL308_Ki', 'CHEMBL2147_Ki']
    library_names = ['specs', 'asinex', 'enamine_hit_locator']

    # Load screening results
    screening_results = pd.read_csv(os.path.join(data_dir, 'all_screening_results.csv'))

    # Subset only molecules from specs. They are the fastest and cheapest for us, so we decided to just go with this
    # single library to minimize screening logistics
    specs_results = screening_results[screening_results['split'] == 'specs']

    # remove molecules that are too similar
    print(f"Removing molecules with a Tanimoto > {TANIMOTO_CUTOFF} ({sum(specs_results['Tanimoto_to_train_max'] > TANIMOTO_CUTOFF)} molecules)")
    specs_results = specs_results[specs_results['Tanimoto_to_train_max'] < TANIMOTO_CUTOFF]

    ##### calculate utopia distance #####
    ranked_data = []
    for dataset in datasets:

        specs_subset = specs_results[specs_results['dataset'] == dataset].copy()

        # Least uncertain + Least unfamiliar
        specs_subset['Least_uncertain_least_unfamiliar'] = calc_utopia_dist(specs_subset['y_E'],
                                                                            specs_subset['y_unc'],
                                                                            specs_subset['unfamiliarity'],
                                                                            maximize=[True, False, False])
        specs_subset['Least_uncertain_least_unfamiliar_ranked'] = rankdata(specs_subset['Least_uncertain_least_unfamiliar'],
                                                                           method='min')

        # Least uncertain + Most unfamiliar
        specs_subset['Least_uncertain_most_unfamiliar'] = calc_utopia_dist(specs_subset['y_E'],
                                                                           specs_subset['y_unc'],
                                                                           specs_subset['unfamiliarity'],
                                                                           maximize=[True, False, True])
        specs_subset['Least_uncertain_most_unfamiliar_ranked'] = rankdata(specs_subset['Least_uncertain_most_unfamiliar'],
                                                                          method='min')

        # Most uncertain + Least unfamiliar
        specs_subset['Most_uncertain_least_unfamiliar'] = calc_utopia_dist(specs_subset['y_E'],
                                                                           specs_subset['y_unc'],
                                                                           specs_subset['unfamiliarity'],
                                                                           maximize=[True, True, False])
        specs_subset['Most_uncertain_least_unfamiliar_ranked'] = rankdata(specs_subset['Most_uncertain_least_unfamiliar'],
                                                                          method='min')

        # Rank the best molecules
        ranked_cols = [col for col in specs_subset.columns if 'ranked' in col]
        n = 10
        top_n = specs_subset[(specs_subset[ranked_cols] <= n).any(axis=1)]

        # Get the top n molecules and the legends for a plot
        mols_a = smiles_to_mols(top_n.sort_values('Least_uncertain_least_unfamiliar_ranked')[:n]['smiles'])
        legends_a = [f"{r} E: {i: .2f}, Unc: {j: .2f}, Unf: {k: .2f}" for r, i, j, k in zip(
            top_n.sort_values('Least_uncertain_least_unfamiliar_ranked')[:n]['Least_uncertain_least_unfamiliar_ranked'],
            top_n.sort_values('Least_uncertain_least_unfamiliar_ranked')[:n]['y_E'],
            top_n.sort_values('Least_uncertain_least_unfamiliar_ranked')[:n]['y_unc'],
            top_n.sort_values('Least_uncertain_least_unfamiliar_ranked')[:n]['unfamiliarity'])]

        mols_b = smiles_to_mols(top_n.sort_values('Least_uncertain_most_unfamiliar_ranked')[:n]['smiles'])
        legends_b = [f"{r} E: {i: .2f}, Unc: {j: .2f}, Unf: {k: .2f}" for r, i, j, k in zip(
            top_n.sort_values('Least_uncertain_most_unfamiliar_ranked')[:n]['Least_uncertain_most_unfamiliar_ranked'],
            top_n.sort_values('Least_uncertain_most_unfamiliar_ranked')[:n]['y_E'],
            top_n.sort_values('Least_uncertain_most_unfamiliar_ranked')[:n]['y_unc'],
            top_n.sort_values('Least_uncertain_most_unfamiliar_ranked')[:n]['unfamiliarity'])]

        mols_c = smiles_to_mols(top_n.sort_values('Most_uncertain_least_unfamiliar_ranked')[:n]['smiles'])
        legends_c = [f"{r} E: {i: .2f}, Unc: {j: .2f}, Unf: {k: .2f}" for r, i, j, k in zip(
            top_n.sort_values('Most_uncertain_least_unfamiliar_ranked')[:n]['Most_uncertain_least_unfamiliar_ranked'],
            top_n.sort_values('Most_uncertain_least_unfamiliar_ranked')[:n]['y_E'],
            top_n.sort_values('Most_uncertain_least_unfamiliar_ranked')[:n]['y_unc'],
            top_n.sort_values('Most_uncertain_least_unfamiliar_ranked')[:n]['unfamiliarity'])]

        all_mols = column_major_reorder(mols_a + mols_b + mols_c, n)  # reorder from [a a a b b b] to [a b a b a b]
        all_legends = column_major_reorder(legends_a + legends_b + legends_c, n)

        plot_molecules_acs1996_grid(all_mols,
                                    subImgSize=(400, 300),
                                    molsPerRow=3,
                                    legends=all_legends
                                    ).save(ospj(data_dir, f'{dataset}_specs_top_n.png'), dpi=(500, 500))

        specs_subset.to_csv(ospj(data_dir, f'{dataset}_specs_ranked.csv'))
        top_n.to_csv(ospj(data_dir, f'{dataset}_specs_top_n.csv'))

