
import os
from os.path import join as ospj
import pandas as pd
import numpy
from constants import ROOTDIR

from cheminformatics.descriptors import mols_to_ecfp
from cheminformatics.utils import smiles_to_mols
from rdkit.DataStructs import BulkTanimotoSimilarity
from cheminformatics.utils import get_scaffold
from rdkit import Chem
from cheminformatics.eval import tani_sim_to_train, substructure_sim_to_train
import numpy as np
import shutil
from warnings import warn


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


def combine_results() -> pd.DataFrame:

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

            dataframes.append(_df)
        except:
            warn(f"Failed loading {filename}")

    # combine dataframe
    df = pd.concat(dataframes)

    return df



def compute_distances(df) -> pd.DataFrame:
    pass


if __name__ == '__main__':

    RESULTS = 'results'

    # Move to root dir
    os.chdir(ROOTDIR)

    # Get the results from the experiments that were run locally (ecfp/cats + random forest)
    get_local_results()

    # Put all results in one big file
    df = combine_results()

    # Compute distance metrics
    df = compute_distances(df)
        # 1. Tanimoto
        # 2. MCS
    #
    #
    # data_path = "results/ecfp_random_forest/CHEMBL4792_Ki/results_preds.csv"
    # df = pd.read_csv(data_path)
    #
    # df.columns
    #
    # train_smiles = list(set(df[df['split'] == 'train'].smiles.tolist()))
    # smiles = list(set(df[df['split'] == 'ood'].smiles.tolist()))
    #
    # Tfull = tani_sim_to_train(smiles, train_smiles, scaffold=False)
    # Tscaf = tani_sim_to_train(smiles, train_smiles, scaffold=True)
    #
    # Sfull = substructure_sim_to_train(smiles, train_smiles, scaffold=False)
    # Sscaf = substructure_sim_to_train(smiles, train_smiles, scaffold=True)
    #
    # from scipy.stats import pearsonr
    #
    # pearsonr(Tfull, Sfull)
    #
    # paracetamol = "CC(=O)Nc1ccc(O)cc1"
    # ibuprofen = "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"
    #
    # paracetamol = Chem.MolFromSmiles(paracetamol)
    # ibuprofen = Chem.MolFromSmiles(ibuprofen)
    #
    # MCSS = MCSSimilarity()
    # MCSS.calc_similarity(paracetamol, ibuprofen, symmetric=False)
    #
    #
    #
    #
    #
    #
    #
    # results = []
    # for dataset in datasets:
    #     df = pd.read_csv(ospj(RESULTS, 'ecfp_random_forest', dataset, 'results_metrics.csv'))
    #     df['dataset'] = dataset
    #     df['descriptor'] = 'ecfp'
    #     results.append(df)
    #
    #     df = pd.read_csv(ospj(RESULTS, 'cats_random_forest', dataset, 'results_metrics.csv'))
    #     df['dataset'] = dataset
    #     df['descriptor'] = 'cats'
    #     results.append(df)
    #
    #
    # results = pd.concat(results)
    #
    # # Group by 'GroupID' and calculate mean and standard deviation
    # results = results.groupby(['dataset', 'descriptor']).agg(['mean', 'std'])
    #
    # results.to_csv(ospj(RESULTS, 'processed', 'random_forest.csv'))
    #
    #
    #
    #
