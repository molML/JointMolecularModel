""" Perform hyperparameter tuning and model training for a MLP control model

Derek van Tilborg
Eindhoven University of Technology
July 2024
"""

import os
from os.path import join as ospj
import copy
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from jcm.config import Config, load_settings, save_settings
from jcm.datasets import load_datasets
from jcm.models import MLP
from jcm.utils import logits_to_pred, prep_outdir, get_all_datasets
from constants import ROOTDIR
from jcm.training import Trainer
from jcm.callbacks import mlp_callback
from sklearn.model_selection import ParameterGrid
from collections import defaultdict


def train_model(config, train_dataset, val_dataset):

    model = MLP(config)

    T = Trainer(config, model, train_dataset, val_dataset)
    if val_dataset is not None:
        T.set_callback('on_batch_end', mlp_callback)
    T.run()

    return model, T


def grid_search(hyperparam_grid: dict[list], config):

    history = defaultdict(list)
    for hypers in ParameterGrid(hyperparam_grid):
        # break
        config_ = copy.copy(config)
        config_.merge_from_dict(hypers)

        n = config_.n_cross_validate
        seeds = np.random.default_rng(seed=config_.random_state).integers(0, 1000, n)

        all_val_losses = []
        for seed in seeds:
            # take a fold from the dataset
            train_dataset, val_dataset, test_dataset, ood_dataset = load_datasets(config_, val_size=config_.val_size,
                                                                                  random_state=seed)
            # train a model
            model, trainer = train_model(config_, train_dataset, val_dataset)
            # add the lowest validation loss to the list of all_val_losses
            all_val_losses.append(min(trainer.history['val_loss']))

        # take the mean over the n folds and add it to the history
        history['val_loss'].append(sum(all_val_losses)/len(all_val_losses))
        history['hypers'].append(hypers)

    return history


def hyperparam_tuning(dataset_name: str, hyper_grid: dict[list]) -> dict:
    """ Perform RF hyperparameter tuning using grid search

    :param dataset_name: name of the dataset (see /data/split)
    :param hyper_grid: dict of hyperparameter options
    :return: best hyperparams
    """

    DEFAULT_SETTINGS_PATH = "experiments/hyperparams/mlp_default.yml"

    experiment_settings = load_settings(DEFAULT_SETTINGS_PATH)
    default_config_dict = experiment_settings['training_config']
    default_config_dict['dataset_name'] = dataset_name
    default_config_dict['out_path'] = None
    default_hyperparameters = experiment_settings['hyperparameters']
    default_hyperparameters['mlp_n_ensemble'] = 1

    config = Config(**default_config_dict)
    config.set_hyperparameters(**default_hyperparameters)

    # Setup the grid search
    grid_search_history = grid_search(hyper_grid, config)

    # Fit the grid search to the data
    print("Starting hyperparameter tuning")
    best_hypers = grid_search_history['hypers'][np.argmin(grid_search_history['val_loss'])]

    # Print the best parameters
    print("Best parameters found: ", best_hypers)
    print("Best cross-validation score: ", min(grid_search_history['val_loss']))

    return best_hypers


def cross_validate(config):
    """

    :param config:
    :return:
    """

    n = config.n_cross_validate
    val_size = config.val_size
    seeds = np.random.default_rng(seed=config.random_state).integers(0, 1000, n)
    out_path = ospj(config.out_path, config.experiment_name, config.dataset_name)
    config.out_path = None

    results = []
    metrics = []
    for seed in seeds:
        # split a chunk of the train data, we don't use the validation data in the RF approach, but we perform cross-
        # validation using the same strategy so we can directly compare methods.
        train_dataset, val_dataset, test_dataset, ood_dataset = load_datasets(config, val_size=val_size, random_state=seed)

        # train model and pickle it afterwards
        model, trainer = train_model(config, train_dataset, val_dataset)

        torch.save(model, ospj(out_path, f"model_{seed}.pt"))

        # perform predictions on all splits
        logits_N_K_C_train, _, y_train = model.predict(train_dataset)
        logits_N_K_C_test, _, y_test = model.predict(test_dataset)
        logits_N_K_C_ood, _, y_ood = model.predict(ood_dataset)

        y_hat_train, y_unc_train = logits_to_pred(logits_N_K_C_train, return_binary=True)
        y_hat_test, y_unc_test = logits_to_pred(logits_N_K_C_test, return_binary=True)
        y_hat_ood, y_unc_ood = logits_to_pred(logits_N_K_C_ood, return_binary=True)

        # Put the predictions in a dataframe
        train_results_df = pd.DataFrame({'seed': seed, 'split': 'train', 'smiles': train_dataset.smiles,
                                         'y': y_train, 'y_hat': y_hat_train, 'y_unc': y_unc_train})
        test_results_df = pd.DataFrame({'seed': seed, 'split': 'test', 'smiles': test_dataset.smiles,
                                        'y': y_test, 'y_hat': y_hat_test, 'y_unc': y_unc_test})
        ood_results_df = pd.DataFrame({'seed': seed, 'split': 'ood', 'smiles': ood_dataset.smiles,
                                       'y': y_ood, 'y_hat': y_hat_ood, 'y_unc': y_unc_ood})
        results_df = pd.concat((train_results_df, test_results_df, ood_results_df))
        results.append(results_df)

        # Put the performance metrics in a dataframe
        metrics.append({'seed': seed,
                        'train_balanced_acc': balanced_accuracy_score(train_dataset.y, y_hat_train),
                        'train_mean_uncertainty': torch.mean(y_unc_train).item(),
                        'test_balanced_acc': balanced_accuracy_score(test_dataset.y, y_hat_test),
                        'test_mean_uncertainty': torch.mean(y_unc_test).item(),
                        'ood_balanced_acc': balanced_accuracy_score(ood_dataset.y, y_hat_ood),
                        'ood_mean_uncertainty': torch.mean(y_unc_ood).item()
                        })

        # log the results/metrics
        pd.concat(results).to_csv(ospj(out_path, 'results_preds.csv'), index=False)
        pd.DataFrame(metrics).to_csv(ospj(out_path, 'results_metrics.csv'), index=False)


if __name__ == '__main__':

    EXPERIMENT_NAME = 'ecfp_mlp'
    DEFAULT_SETTINGS_PATH = "experiments/hyperparams/mlp_default.yml"
    HYPERPARAM_GRID = {'mlp_hidden_dim': [1024, 2048],
                       'mlp_n_layers': [1, 2, 3],
                       'lr': [3e-4, 3e-5, 3e-6]}

    # move to root dir
    os.chdir(ROOTDIR)

    all_datasets = get_all_datasets()
    for dataset_name in tqdm(all_datasets):
        print(dataset_name)

        best_hypers = hyperparam_tuning(dataset_name, HYPERPARAM_GRID)

        settings = load_settings(DEFAULT_SETTINGS_PATH)
        config_dict = settings['training_config'] | {'dataset_name': dataset_name, 'experiment_name': EXPERIMENT_NAME}
        hyperparameters = settings['hyperparameters'] | best_hypers

        config = Config(**config_dict)
        config.set_hyperparameters(**hyperparameters)

        # make output dir
        prep_outdir(config)

        # save best hypers
        save_settings(config, ospj(config.out_path, config.experiment_name, config.dataset_name, 'experiment_settings.yml'))

        # perform model training with cross validation and save results
        results = cross_validate(config)
