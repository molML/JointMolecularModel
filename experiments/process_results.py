
import os
from os.path import join as ospj
import pandas as pd
from constants import ROOTDIR
from cheminformatics.multiprocessing import tani_sim_to_train, substructure_sim_to_train
import shutil
from warnings import warn
from collections import defaultdict
from cheminformatics.cleaning import canonicalize_smiles


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


def process_results() -> pd.DataFrame:

    all_results_path = ospj(RESULTS, 'all_results')
    files = [i for i in os.listdir(all_results_path) if not i.startswith('.')]

    dataframes = []
    for filename in files:
        try:
            # parse the filename
            descriptor, model_type, dataset_name = filename.replace('random_forest', 'rf').split('_', maxsplit=2)
            dataset_name = '_'.join(dataset_name.split('_')[:-2])

            # read df and add info from filename
            _df = pd.read_csv(ospj(all_results_path, filename))
            _df['descriptor'] = descriptor
            _df['model_type'] = model_type
            _df['dataset_name'] = dataset_name

            print(f'Canonicalizing {len(_df)} SMILES.')
            _df['smiles'] = canonicalize_smiles(_df.smiles)

            print('Computing distance/similarity metrics.')
            metrics = compute_metrics(_df)
            _df = _df.assign(**metrics)

            _df.to_csv(ospj(RESULTS, 'processed', filename.replace('_results_preds', '_processed')), index=False)

            dataframes.append(_df)
        except:
            warn(f"Failed loading {filename}")

    # combine dataframe
    df = pd.concat(dataframes)

    return df


def compute_metrics(df):
    # for each dataset
    #  1. find train molecules
    #  2. get distance of every molecule to the train set

    train_smiles = list(set(df[df['split'] == 'train'].smiles))
    all_smiles = df.smiles.tolist()
    all_unique_smiles = list(set(all_smiles))

    # since smiles occur n times in a dataset, we only compute the distances for the unique ones and map them back
    tani_sim = tani_sim_to_train(all_unique_smiles, train_smiles)
    substructure_sim = substructure_sim_to_train(all_unique_smiles, train_smiles)

    metrics = {}
    for i in range(len(all_unique_smiles)):
        metrics[all_unique_smiles[i]] = {'tanimoto': tani_sim[i],
                                         'substructure_sim': substructure_sim[i]}

    # map every value back to the original one
    matched_metrics = defaultdict(list)
    for smi in all_smiles:
        matched_metrics['smiles2'].append(smi)
        matched_metrics['tanimoto'].append(metrics[smi]['tanimoto'])
        matched_metrics['substructure_sim'].append(metrics[smi]['substructure_sim'])

    return matched_metrics


if __name__ == '__main__':

    RESULTS = 'results'

    # Move to root dir
    os.chdir(ROOTDIR)

    # Get the results from the experiments that were run locally (ecfp/cats + random forest)
    get_local_results()

    # Put all results in one big file
    df = process_results()

    df.to_csv(ospj(RESULTS, 'processed', 'all_results_processed.csv'), index=False)
