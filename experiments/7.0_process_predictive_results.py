""" Merge and process all results from previous experiments

Derek van Tilborg
Eindhoven University of Technology
November 2024
"""


import os
from os.path import join as ospj
import pandas as pd
from constants import ROOTDIR
import shutil
from warnings import warn
from collections import Counter
from sklearn.metrics import confusion_matrix

RESULTS = 'results'


def get_local_results() -> None:
    """ Get the results from the experiments that were run locally and move them to 'all_results """

    experiments = ['cats_random_forest', 'ecfp_random_forest']
    out_path = ospj(RESULTS, 'all_results')

    for experi in experiments:
        datasets = [i for i in os.listdir(ospj(RESULTS, experi)) if not i.startswith('.')]
        for dataset in datasets:
            src = ospj(RESULTS, experi, dataset, 'results_preds.csv')
            dst = ospj(out_path, f"{experi}_{dataset}_results_preds.csv")
            shutil.copyfile(src, dst)


def combine_all_results() -> pd.DataFrame:
    """ Combines all results from the 'all_results' directory into one big file and add precomputed distance metrics
    to all rows """

    all_results_path = ospj(RESULTS, 'all_results')
    files = [i for i in os.listdir(all_results_path) if not i.startswith('.')]

    # precomputed distance metrics for all datasets
    df_metrics = pd.read_csv("data/datasets_with_metrics/all_datasets.csv")

    fails = []
    dataframes = []
    for filename in files:
        # break
        try:
            print(f'Parsing {filename}.')

            # parse the filename
            descriptor, model_type, dataset_name = filename.replace('random_forest', 'rf').split('_', maxsplit=2)
            dataset_name = '_'.join(dataset_name.split('_')[:-2])

            # read df and add info from filename
            df_results = pd.read_csv(ospj(all_results_path, filename))
            df_results['descriptor'] = descriptor
            df_results['model_type'] = model_type
            df_results['dataset'] = dataset_name
            df_results['split_TPR'] = None
            df_results['split_TNR'] = None
            df_results['split_acc'] = None
            df_results['split_balanced_acc'] = None

            # compute performance metrics for the whole split
            if 'y' in df_results.columns:
                for split in set(df_results['split']):

                    average_predictions_over_folds = df_results[df_results['split'] == split].groupby(['smiles', 'y'])['y_hat'].mean().reset_index()

                    y = average_predictions_over_folds['y']
                    y_hat = average_predictions_over_folds['y_hat']
                    y_hat = 1*(y_hat >= 0.5 )

                    tn, fp, fn, tp = confusion_matrix(y, y_hat).ravel()
                    tpr = tp / (tp + fn)  # recall/sensitivity
                    tnr = tn / (tn + fp)  # specificity/selectivity
                    acc = (tp + tn) / (tp + fn + tn + fp)
                    balanced_acc = (tpr + tnr) / 2

                    df_results.loc[df_results['split'] == split, 'split_TPR'] = tpr
                    df_results.loc[df_results['split'] == split, 'split_TNR'] = tnr
                    df_results.loc[df_results['split'] == split, 'split_acc'] = acc
                    df_results.loc[df_results['split'] == split, 'split_balanced_acc'] = balanced_acc

            # Add distance metrics to the results dataframe
            df_metrics_subset = df_metrics[df_metrics['dataset'] == dataset_name]

            columns_to_keep = ['smiles'] + [col for col in df_metrics_subset.columns if col not in df_results.columns]
            df_metrics_subset = df_metrics_subset.loc[:, columns_to_keep]  # get rid of duplicate columns
            df_results = pd.merge(df_results, df_metrics_subset, on='smiles', how='left', indicator=True, validate='many_to_one')

            if not set(df_results._merge) == {"both"}:
                warn(f'Imperfect match for {filename}! Matching indicator: {dict(Counter(df_results._merge))}')
                fails.append(filename)

            # save dataframe as csv
            df_results.to_csv(ospj(RESULTS, 'processed', filename.replace('_results_preds', '_processed')), index=False)

            dataframes.append(df_results)
        except Exception as e:
            warn(f"Failed parsing {filename} due to {e}")

    # combine dataframe
    df = pd.concat(dataframes)

    # add column to smiles ID (I can kick out duplicates later to immediately get the summary df)
    df['smiles_id'] = pd.factorize(df['smiles'])[0]

    print(f'Done, but these files had a problem: {fails}')

    return df


if __name__ == '__main__':

    # Move to root dir
    os.chdir(ROOTDIR)

    # Get the results from the experiments that were run locally (ecfp/cats + random forest)
    get_local_results()

    # Put all results in one big file
    all_results = combine_all_results()

    all_results.to_csv(ospj(RESULTS, 'processed', 'all_results_processed.csv'), index=False)
