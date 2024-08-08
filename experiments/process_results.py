
import os
from os.path import join as ospj
import pandas as pd
from constants import ROOTDIR
import shutil
from warnings import warn
from cheminformatics.cleaning import canonicalize_smiles

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

    dataframes = []
    for filename in files:
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

            # canoncincalize smiles
            df_results['smiles'] = canonicalize_smiles(df_results.smiles)

            # Add distance metrics to the results dataframe
            df_metrics_subset = df_metrics[df_metrics['dataset'] == dataset_name]

            columns_to_keep = ['smiles'] + [col for col in df_metrics_subset.columns if col not in df_results.columns]
            df_metrics_subset = df_metrics_subset.loc[:, columns_to_keep]  # get rid of duplicate columns
            df_results = pd.merge(df_results, df_metrics_subset, on='smiles', how='left')

            # save dataframe as csv
            df_results.to_csv(ospj(RESULTS, 'processed', filename.replace('_results_preds', '_processed')), index=False)

            dataframes.append(df_results)
        except:
            warn(f"Failed parsing {filename}")

    # combine dataframe
    df = pd.concat(dataframes)
    print('Done')

    return df


if __name__ == '__main__':

    # Move to root dir
    os.chdir(ROOTDIR)

    # Get the results from the experiments that were run locally (ecfp/cats + random forest)
    get_local_results()

    # Put all results in one big file
    all_results = combine_all_results()

    all_results.to_csv(ospj(RESULTS, 'processed', 'all_results_processed.csv'), index=False)
