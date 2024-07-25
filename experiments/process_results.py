import os
from os.path import join as ospj
import pandas as pd
import numpy
from constants import ROOTDIR

if __name__ == '__main__':

    RESULTS = 'results'

    # move to root dir
    os.chdir(ROOTDIR)

    # RF control expriments
    datasets = [i for i in os.listdir(ospj(RESULTS, 'ecfp_random_forest')) if not i.startswith('.')]

    results = []
    for dataset in datasets:
        df = pd.read_csv(ospj(RESULTS, 'ecfp_random_forest', dataset, 'results_metrics.csv'))
        df['dataset'] = dataset
        df['descriptor'] = 'ecfp'
        results.append(df)

        df = pd.read_csv(ospj(RESULTS, 'cats_random_forest', dataset, 'results_metrics.csv'))
        df['dataset'] = dataset
        df['descriptor'] = 'cats'
        results.append(df)


    results = pd.concat(results)

    # Group by 'GroupID' and calculate mean and standard deviation
    results = results.groupby(['dataset', 'descriptor']).agg(['mean', 'std'])

    results.to_csv(ospj(RESULTS, 'processed', 'random_forest.csv'))