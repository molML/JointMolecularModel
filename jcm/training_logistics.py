""" Contains all functions that are required for model training and hyperparameter tuning

Derek van Tilborg
Eindhoven University of Technology
July 2024
"""

import os
import copy
from os.path import join as ospj
from collections import defaultdict
from typing import Callable
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from jcm.config import Config, load_settings
from jcm.datasets import load_datasets, MoleculeDataset
from jcm.models import RfEnsemble, MLP
from jcm.callbacks import mlp_callback
from jcm.utils import logits_to_pred
from jcm.training import Trainer


def prep_outdir(config: Config):
    """ Create the output directory if needed"""

    outdir = ospj(config.out_path, config.experiment_name, config.dataset_name)
    os.makedirs(outdir, exist_ok=True)


def get_all_datasets() -> list[str]:
    """ Return a list of all dataset names """

    all_datasets = os.listdir(ospj('data', 'split'))
    all_datasets = [i for i in all_datasets if i.endswith(".csv") and i != 'ChEMBL_33_split.csv']
    all_datasets = [i.replace('_split.csv', '') for i in all_datasets]

    return all_datasets


def rf_hyperparam_tuning(dataset_name: str, config_default_path: str, hyper_grid: dict[list]) -> dict:
    """ Perform RF hyperparameter tuning using grid search

    :param dataset_name: name of the dataset (see /data/split)
    :param config_default_path: path of the default config.yml
    :param hyper_grid: dict of hyperparameter options
    :return: best hyperparams
    """

    experiment_settings = load_settings(config_default_path)
    default_config_dict = experiment_settings['training_config']
    default_config_dict['dataset_name'] = dataset_name
    default_hyperparameters = experiment_settings['hyperparameters']

    config = Config(**default_config_dict)
    config.set_hyperparameters(**default_hyperparameters)

    train_dataset, val_dataset, test_dataset, ood_dataset = load_datasets(config, val_size=0)

    # Setup the grid search
    class_weight = "balanced" if config.balance_classes else None
    grid_search = GridSearchCV(estimator=RandomForestClassifier(class_weight=class_weight),
                               param_grid=hyper_grid, cv=config.n_cross_validate, verbose=0, n_jobs=-1)

    # Fit the grid search to the data
    print("Starting hyperparameter tuning")
    grid_search.fit(*train_dataset.xy_np())

    # Print the best parameters
    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation score: ", grid_search.best_score_)

    return grid_search.best_params_


def rf_cross_validate(config: Config):
    """ Performs cross-validation of a random forest model using the settings in the config class

    :param config: config class
    """

    n = config.n_cross_validate
    val_size = config.val_size
    seeds = np.random.default_rng(seed=config.random_state).integers(0, 1000, n)
    out_path = ospj(config.out_path, config.experiment_name, config.dataset_name)

    results = []
    metrics = []
    for seed in seeds:
        # split a chunk of the train data, we don't use the validation data in the RF approach, but we perform cross-
        # validation using the same strategy so we can directly compare methods.
        train_dataset, _, test_dataset, ood_dataset = load_datasets(config, val_size=val_size, random_state=seed)
        x_train, y_train = train_dataset.xy_np()
        x_test, y_test = test_dataset.xy_np()
        x_ood, y_ood = ood_dataset.xy_np()

        # train model and pickle it afterwards
        model = RfEnsemble(config)
        model.train(x_train, y_train)
        torch.save(model, ospj(out_path, f"model_{seed}.pt"))

        # perform predictions on all splits
        logits_N_K_C_train = model.predict(x_train)
        logits_N_K_C_test = model.predict(x_test)
        logits_N_K_C_ood = model.predict(x_ood)

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
                        'train_balanced_acc': balanced_accuracy_score(y_train, y_hat_train),
                        'train_mean_uncertainty': torch.mean(y_unc_train).item(),
                        'test_balanced_acc': balanced_accuracy_score(y_test, y_hat_test),
                        'test_mean_uncertainty': torch.mean(y_unc_test).item(),
                        'ood_balanced_acc': balanced_accuracy_score(y_ood, y_hat_ood),
                        'ood_mean_uncertainty': torch.mean(y_unc_ood).item()
                        })

        # log the results/metrics
        pd.concat(results).to_csv(ospj(out_path, 'results_preds.csv'), index=False)
        pd.DataFrame(metrics).to_csv(ospj(out_path, 'results_metrics.csv'), index=False)


def train_model(model: Callable, callback: Callable, config: Config, train_dataset: MoleculeDataset,
                val_dataset: MoleculeDataset):
    """ Train a model according to its config

    :param model: model class from jcm.models
    :param callback: callback fuction from jcm.callbacks
    :param config: config class
    :param train_dataset: training dataset
    :param val_dataset: validation dataset (is required for callbacks)
    :return: trained model, trainer
    """

    f = model(config)

    T = Trainer(config, f, train_dataset, val_dataset)
    if val_dataset is not None:
        T.set_callback('on_batch_end', callback)
    T.run()

    return f, T


def nn_grid_search(hyperparam_grid: dict[str, list], config: Config):

    history = defaultdict(list)
    for hypers in ParameterGrid(hyperparam_grid):

        config_ = copy.copy(config)
        updated_hypers = config_.hyperparameters | hypers
        config_.set_hyperparameters(**updated_hypers)

        n = config_.n_cross_validate
        seeds = np.random.default_rng(seed=config_.random_state).integers(0, 1000, n)

        all_val_losses = []
        for seed in seeds:
            # take a fold from the dataset
            train_dataset, val_dataset, test_dataset, ood_dataset = load_datasets(config_, val_size=config_.val_size,
                                                                                  random_state=seed)
            # train a model
            model, trainer = train_model(MLP, mlp_callback, config_, train_dataset, val_dataset)
            # add the lowest validation loss to the list of all_val_losses
            all_val_losses.append(min(trainer.history['val_loss']))

        # take the mean over the n folds and add it to the history
        history['val_loss'].append(sum(all_val_losses)/len(all_val_losses))
        history['hypers'].append(hypers)

    return history


def nn_hyperparam_tuning(dataset_name: str, default_config_path: str, hyper_grid: dict[str, list]) -> dict:
    """ Perform RF hyperparameter tuning using grid search

    :param dataset_name: name of the dataset (see /data/split)
    :param default_config_path: path of the default config
    :param hyper_grid: dict of hyperparameter options
    :return: best hyperparams
    """

    experiment_settings = load_settings(default_config_path)
    default_config_dict = experiment_settings['training_config']
    default_config_dict['dataset_name'] = dataset_name
    default_config_dict['out_path'] = None
    default_hyperparameters = experiment_settings['hyperparameters']
    default_hyperparameters['mlp_n_ensemble'] = 1

    config = Config(**default_config_dict)
    config.set_hyperparameters(**default_hyperparameters)

    # Setup the grid search
    grid_search_history = nn_grid_search(hyper_grid, config)

    # Fit the grid search to the data
    print("Starting hyperparameter tuning")
    best_hypers = grid_search_history['hypers'][np.argmin(grid_search_history['val_loss'])]

    # Print the best parameters
    print("Best parameters found: ", best_hypers)
    print("Best cross-validation score: ", min(grid_search_history['val_loss']))

    return best_hypers


def nn_cross_validate(config: Config):
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
        model, trainer = train_model(MLP, mlp_callback, config, train_dataset, val_dataset)

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
                                         'y': y_train.cpu(), 'y_hat': y_hat_train.cpu(), 'y_unc': y_unc_train.cpu()})
        test_results_df = pd.DataFrame({'seed': seed, 'split': 'test', 'smiles': test_dataset.smiles,
                                        'y': y_test.cpu(), 'y_hat': y_hat_test.cpu(), 'y_unc': y_unc_test.cpu()})
        ood_results_df = pd.DataFrame({'seed': seed, 'split': 'ood', 'smiles': ood_dataset.smiles,
                                       'y': y_ood.cpu(), 'y_hat': y_hat_ood.cpu(), 'y_unc': y_unc_ood.cpu()})
        results_df = pd.concat((train_results_df, test_results_df, ood_results_df))
        results.append(results_df)

        # Put the performance metrics in a dataframe
        metrics.append({'seed': seed,
                        'train_balanced_acc': balanced_accuracy_score(train_dataset.y.cpu(), y_hat_train.cpu()),
                        'train_mean_uncertainty': torch.mean(y_unc_train).item(),
                        'test_balanced_acc': balanced_accuracy_score(test_dataset.y.cpu(), y_hat_test.cpu()),
                        'test_mean_uncertainty': torch.mean(y_unc_test).item(),
                        'ood_balanced_acc': balanced_accuracy_score(ood_dataset.y.cpu(), y_hat_ood.cpu()),
                        'ood_mean_uncertainty': torch.mean(y_unc_ood).item()
                        })

        # log the results/metrics
        pd.concat(results).to_csv(ospj(out_path, 'results_preds.csv'), index=False)
        pd.DataFrame(metrics).to_csv(ospj(out_path, 'results_metrics.csv'), index=False)
