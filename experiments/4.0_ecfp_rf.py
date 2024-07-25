""" Perform hyperparameter tuning and model training for a Random Forest control model

Derek van Tilborg
Eindhoven University of Technology
July 2024
"""

import os
from os.path import join as ospj
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from jcm.config import Config, load_settings, save_settings
from jcm.datasets import load_datasets
from jcm.models import RfEnsemble
from jcm.utils import logits_to_pred, prep_outdir, get_all_datasets
from constants import ROOTDIR


def hyperparam_tuning(dataset_name: str, hyper_grid: dict[list]) -> dict:
    """ Perform RF hyperparameter tuning using grid search

    :param dataset_name: name of the dataset (see /data/split)
    :param hyper_grid: dict of hyperparameter options
    :return: best hyperparams
    """

    DEFAULT_SETTINGS_PATH = "experiments/hyperparams/rf_default.yml"

    experiment_settings = load_settings(DEFAULT_SETTINGS_PATH)
    default_config_dict = experiment_settings['training_config']
    default_config_dict['descriptor'] = 'ecfp'
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


def cross_validate(config):
    """

    :param config:
    :return:
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


if __name__ == '__main__':

    EXPERIMENT_NAME = 'ecfp_random_forest'
    DEFAULT_SETTINGS_PATH = "experiments/hyperparams/rf_default.yml"
    HYPERPARAM_GRID = {'n_estimators': [100, 250, 500, 1000],
                           'max_depth': [None, 10, 20, 30],
                           'min_samples_split': [2, 5, 10]}

    # move to root dir
    os.chdir(ROOTDIR)

    all_datasets = get_all_datasets()
    for dataset_name in tqdm(all_datasets):
        print(dataset_name)

        best_hypers = hyperparam_tuning(dataset_name, HYPERPARAM_GRID)

        settings = load_settings(DEFAULT_SETTINGS_PATH)
        config_dict = settings['training_config'] | {'dataset_name': dataset_name, 'experiment_name': EXPERIMENT_NAME, 'descriptor': 'ecfp'}
        hyperparameters = settings['hyperparameters'] | best_hypers

        config = Config(**config_dict)
        config.set_hyperparameters(**hyperparameters)

        # make output dir
        prep_outdir(config)

        # save best hypers
        save_settings(config, ospj(config.out_path, config.experiment_name, config.dataset_name, 'experiment_settings.yml'))

        # perform model training with cross validation and save results
        results = cross_validate(config)
