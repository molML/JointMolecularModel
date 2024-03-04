
import os
import itertools
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import torch
from jcm.utils import predict_and_eval_mlp
from jcm.datasets import load_moleculeace
from jcm.trainer import train_mlp
from jcm.config import Config
from constants import ROOTDIR


OUT_DIR = os.path.join(ROOTDIR, 'results/pre_trained_mlps')
OUT_FILE_HYPERTUNING = 'mlp_hyper_tuning.csv'
OUT_FILE_RESULTS = 'mlp_pretraining_results.csv'
N_FOLDS = 5
VAL_SPLIT = 0.2
MAX_ITERS = 100
BATCH_SIZE = 128

DEFAULT_HYPERPARAMETERS = {"input_dim": 1024, "hidden_dim": 1024, "output_dim": 2, "anchored": True, "l2_lambda": 3e-4,
                           "n_ensemble": 10}

n_layers = [1, 2, 3]
lr = [3e-3, 3e-4, 3e-5]
TUNEABLE_HYPERPARAMETERS = [{'n_layers': i[0], 'lr': i[1]} for i in itertools.product(n_layers, lr)]


if __name__ == '__main__':

    datasets = [f'data/moleculeace/{i}' for i in os.listdir('data/moleculeace') if i.startswith('CHEMBL')]
    history = defaultdict(list)

    for dataset in tqdm(datasets):

        for hypers in TUNEABLE_HYPERPARAMETERS:
            for seed in range(N_FOLDS):
                try:
                    # get the data splits
                    train_dataset, val_dataset, test_dataset = load_moleculeace(dataset, val_split=VAL_SPLIT, seed=seed)

                    # add the hypers of this iteration to the default hypers
                    hyperparameters = DEFAULT_HYPERPARAMETERS | hypers

                    # init config
                    config = Config(max_iters=MAX_ITERS, batch_size=BATCH_SIZE, batch_end_callback_every=MAX_ITERS, out_path=None)
                    config.set_hyperparameters(**hyperparameters)

                    # train model
                    model, trainer = train_mlp(config, train_dataset, val_dataset)

                    # calculate metrics
                    metrics = predict_and_eval_mlp(model, val_dataset)

                    history['dataset'].append(dataset.split('/')[-1].split('.csv')[0])
                    for hyper_name, val in hypers.items():
                        history[hyper_name].append(val)
                    history['seed'].append(seed)
                    history['BA'].append(metrics['BA'])

                    df = pd.DataFrame(history)
                    df.to_csv(os.path.join(OUT_DIR, OUT_FILE_HYPERTUNING))
                except:
                    print(f"Failed {dataset} {hypers} {seed}")

    # find the best hypers for each dataset using some pandas magic
    hyper_results = pd.read_csv(os.path.join(OUT_DIR, OUT_FILE_HYPERTUNING), index_col=False).iloc[:, 1:]
    hyper_results = hyper_results.groupby(['dataset', 'n_layers', 'lr']).agg({'BA': ['mean', 'std']}).reset_index()
    hyper_results.columns = [''.join(col).strip() for col in hyper_results.columns.values]
    best_hypers = hyper_results.loc[hyper_results.groupby('dataset')['BAmean'].idxmax()]

    # Train all models using the best hypers using all training data. Evaluate on test set
    results = []
    for dataset_name in tqdm(best_hypers.dataset):

        n_layers = best_hypers[best_hypers['dataset'] == dataset_name].n_layers.item()
        lr = best_hypers[best_hypers['dataset'] == dataset_name].lr.item()

        # get the data splits
        train_dataset, val_dataset, test_dataset = load_moleculeace(f"data/moleculeace/{dataset_name}.csv", val_split=0)

        # add the hypers of this iteration to the default hypers
        hyperparameters = DEFAULT_HYPERPARAMETERS | {'n_layers': n_layers, 'lr': lr}

        # init config
        config = Config(max_iters=MAX_ITERS, batch_size=BATCH_SIZE, out_path=None)
        config.set_hyperparameters(**hyperparameters)

        # train model
        model, trainer = train_mlp(config, train_dataset)

        # save model
        torch.save(model, os.path.join(OUT_DIR, f"MLP_{dataset_name}.pt"))

        # calculate metrics
        metrics = predict_and_eval_mlp(model, test_dataset)
        results.append(metrics)

    # write file with all results
    pd.concat([best_hypers.reset_index(), pd.DataFrame(results)], axis=1).to_csv(os.path.join(OUT_DIR, OUT_FILE_RESULTS))
