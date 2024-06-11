
import os
import itertools
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import torch
from jcm.utils import predict_and_eval_mlp
from jcm.datasets import load_moleculeace
from jcm.training import train_mlp
from jcm.config import Config
from constants import ROOTDIR


OUT_DIR = os.path.join(ROOTDIR, 'results/pre_trained_mlps')
OUT_FILE_HYPERTUNING = 'mlp_hyper_tuning.csv'
OUT_FILE_RESULTS = 'mlp_pretraining_results.csv'
N_FOLDS = 10
VAL_SPLIT = 0.2
MAX_ITERS = 100
BATCH_SIZE = 128

DEFAULT_HYPERPARAMETERS = {"input_dim": 2048, "hidden_dim": 2048, "output_dim": 2, "anchored": True, "l2_lambda": 3e-4,
                           "n_ensemble": 10, 'latent_dim': 64}

n_layers = [1, 2, 3]
lr = [3e-3, 3e-4, 3e-5]
TUNEABLE_HYPERPARAMETERS = [{'n_layers': i[0], 'lr': i[1]} for i in itertools.product(n_layers, lr)]


def hyper_param_tuning(dataset, hyperparameters: list[dict], folds: int = 10, val_split: float = 0.2,
                       max_iters: int = 100, batch_size: int = 100):
    scores = []
    for hypers in tqdm(hyperparameters):
        seed_score = []
        for seed in range(folds):
            try:
                # get the data splits
                train_dataset, val_dataset, test_dataset = load_moleculeace(dataset, val_split=val_split, seed=seed)

                # add the hypers of this iteration to the default hypers
                _hyperparameters = DEFAULT_HYPERPARAMETERS | hypers

                # init config
                config = Config(max_iters=max_iters, batch_size=batch_size, batch_end_callback_every=None, out_path=None)
                config.set_hyperparameters(**_hyperparameters)

                # train model
                model, trainer = train_mlp(config, train_dataset, val_dataset)

                # calculate metrics
                metrics = predict_and_eval_mlp(model, val_dataset)
                seed_score.append(metrics['BA'])

            except:
                print(f"Failed {dataset} {hypers} {seed}")

        scores.append(sum(seed_score) / len(seed_score))

    return hyperparameters[scores.index(max(scores))]


def cross_validate(dataset, best_hypers, folds: int = 10, val_split: float = 0.2, save_path: str = None,
                   max_iters: int = 100, batch_size: int = 100):

    dataset_name = dataset.split('/')[-1].split('.csv')[0]
    results = defaultdict(list)

    # add the hypers of this iteration to the default hypers
    _hyperparameters = DEFAULT_HYPERPARAMETERS | best_hypers

    # init config
    config = Config(max_iters=max_iters, batch_size=batch_size, batch_end_callback_every=None, out_path=None)
    config.set_hyperparameters(**_hyperparameters)

    for seed in range(folds):

        train_dataset, val_dataset, test_dataset = load_moleculeace(dataset, val_split=val_split, seed=seed)

        # train model
        model, trainer = train_mlp(config, train_dataset, val_dataset)

        val_metrics = predict_and_eval_mlp(model, val_dataset)
        val_metrics = {'val_' + k: v for k, v in val_metrics.items()}

        test_metrics = predict_and_eval_mlp(model, test_dataset)
        test_metrics = {'test_' + k: v for k, v in test_metrics.items()}

        results_fold = {'dataset': dataset_name, 'fold': seed} | val_metrics | test_metrics
        for k, v in results_fold.items():
            results[k].append(v)

    # train final model
    train_dataset, val_dataset, test_dataset = load_moleculeace(dataset, val_split=0)

    # train model
    model, trainer = train_mlp(config, train_dataset, val_dataset)

    if save_path is not None:
        pd.DataFrame(results).to_csv(os.path.join(save_path, f"{dataset_name}_mlp_results.csv"))
        torch.save(model, os.path.join(save_path, f"{dataset_name}_mlp.pt"))

    return model


if __name__ == '__main__':

    datasets = [f'data/moleculeace/{i}' for i in os.listdir('data/moleculeace') if i.startswith('CHEMBL')]
    hyper_dict = {}

    for dataset in datasets:
        dataset_name = dataset.split('/')[-1].split('.csv')[0]

        best_hypers = hyper_param_tuning(dataset, TUNEABLE_HYPERPARAMETERS, folds=N_FOLDS, max_iters=MAX_ITERS,
                                         batch_size=BATCH_SIZE, val_split=VAL_SPLIT)

        hyper_dict[dataset_name] = best_hypers
        torch.save(hyper_dict, 'results/best_hyperparameters.pt')

        model = cross_validate(dataset, best_hypers, folds=N_FOLDS, max_iters=MAX_ITERS, batch_size=BATCH_SIZE,
                               val_split=VAL_SPLIT, save_path='results/pre_trained_mlps')




