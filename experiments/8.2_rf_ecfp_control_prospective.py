""" Perform hyperparameter tuning and model training for a Random Forest control model

Derek van Tilborg
Eindhoven University of Technology
March 2025
"""

import os
import copy
import torch
import pandas as pd
import numpy as np
from os.path import join as ospj
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from jcm.training_logistics import prep_outdir
from jcm.datasets import MoleculeDataset
from jcm.models import RfEnsemble
from jcm.config import Config, load_settings, save_settings
from jcm.utils import logits_to_pred
from constants import ROOTDIR


def load_full_data(config, val_size: float = None, seed: int = 0 , **kwargs):

    config_ = copy.copy(config)
    updated_hypers = config_.hyperparameters | kwargs
    config_.set_hyperparameters(**updated_hypers)

    data_path = ospj(f'data/clean/{config_.dataset_name}.csv')

    # get the train and val SMILES from the pre-processed file
    train_data = pd.read_csv(data_path)

    if val_size is not None:
        train_data, val_data = train_test_split(train_data, test_size=config_.val_size, random_state=seed)
        val_dataset = MoleculeDataset(val_data.smiles.tolist(), val_data.y.tolist(),
                                      descriptor=config_.descriptor, randomize_smiles=config_.data_augmentation)
    else:
        val_dataset = None

    train_dataset = MoleculeDataset(train_data.smiles.tolist(), train_data.y.tolist(),
                                   descriptor=config_.descriptor, randomize_smiles=config_.data_augmentation)

    return train_dataset, val_dataset


def generate_mc_splits(X, y, n_splits=5, val_size=0.9, seed=42):
    """ Create seed-determined Monte Carlo splits """
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=val_size, random_state=seed)

    return [(train_idx, val_idx) for train_idx, val_idx in sss.split(X, y)]


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

    train_dataset, val_dataset = load_full_data(config)
    X, Y = train_dataset.xy_np()

    # Get stratified splits
    cv = generate_mc_splits(X, Y, n_splits=config.n_cross_validate_hyperparam,
                            val_size=config.val_size, seed=config.random_state)

    # Setup the grid search
    class_weight = "balanced" if config.balance_classes else None
    grid_search = GridSearchCV(estimator=RandomForestClassifier(class_weight=class_weight),
                               param_grid=hyper_grid, cv=cv, verbose=0, n_jobs=-1)

    # Fit the grid search to the data
    print("Starting hyperparameter tuning")
    grid_search.fit(X, Y)

    # Print the best parameters
    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation score: ", grid_search.best_score_)

    return grid_search.best_params_


def rf_cross_validate_and_inference(config: Config, libraries: None):
    """ Performs cross-validation of a random forest model using the settings in the config class

    :param config: config class
    """

    n = config.n_cross_validate
    seeds = np.random.default_rng(seed=config.random_state).integers(0, 1000, n)

    results = []
    metrics = []
    for seed in seeds:
        # split a chunk of the train data, we don't use the validation data in the RF approach, but we perform cross-
        # validation using the same strategy so we can directly compare methods.

        train_dataset, val_dataset = load_full_data(config, val_size=config.val_size, seed=seed)

        x_train, y_train = train_dataset.xy_np()
        x_val, y_val = val_dataset.xy_np()

        # train model and pickle it afterwards
        model = RfEnsemble(config)
        model.train(x_train, y_train)
        torch.save(model, ospj(config.out_path, f"model_{seed}.pt"))

        logits_N_K_C_train = model.predict(x_train)
        logits_N_K_C_val = model.predict(x_val)

        y_hat_train, y_unc_train = logits_to_pred(logits_N_K_C_train, return_binary=True)
        y_hat_val, y_unc_val = logits_to_pred(logits_N_K_C_val, return_binary=True)

        # Put the predictions in a dataframe
        train_results_df = pd.DataFrame({'seed': seed, 'split': 'train', 'smiles': train_dataset.smiles,
                                         'y': y_train, 'y_hat': y_hat_train, 'y_unc': y_unc_train})
        val_results_df = pd.DataFrame({'seed': seed, 'split': 'test', 'smiles': val_dataset.smiles,
                                        'y': y_val, 'y_hat': y_hat_val, 'y_unc': y_unc_val})

        all_dfs = [train_results_df, val_results_df]
        if libraries is not None:
            for library_name, library in libraries.items():
                print(f"performing inference on {library_name} ({seed})")

                lib_x, _ = library.xy_np()

                # perform predictions on all splits
                logits_N_K_C_lib = model.predict(lib_x)

                y_hat_lib, y_unc_lib = logits_to_pred(logits_N_K_C_lib, return_binary=True)

                lib_results_df = pd.DataFrame({'seed': seed, 'split': library_name, 'smiles': library.smiles,
                                               'y': None, 'y_hat': y_hat_lib, 'y_unc': y_unc_lib})

                all_dfs.append(lib_results_df)

        results_df = pd.concat(all_dfs)
        results.append(results_df)

        # Put the performance metrics in a dataframe
        metrics.append({'seed': seed,
                        'train_balanced_acc': balanced_accuracy_score(y_train, y_hat_train),
                        'train_mean_uncertainty': torch.mean(y_unc_train).item(),
                        'val_balanced_acc': balanced_accuracy_score(y_val, y_hat_val),
                        'val_mean_uncertainty': torch.mean(y_unc_val).item(),
                        })

        # log the results/metrics
        pd.concat(results).to_csv(ospj(config.out_path, 'results_preds.csv'), index=False)
        pd.DataFrame(metrics).to_csv(ospj(config.out_path, 'results_metrics.csv'), index=False)


if __name__ == '__main__':

    DEFAULT_SETTINGS_PATH = "experiments/hyperparams/ecfp_rf_default_prospective.yml"
    HYPERPARAM_GRID = {'n_estimators': [100, 250, 500, 1000],
                           'max_depth': [None, 10, 20, 30],
                           'min_samples_split': [2, 5, 10]}

    datasets = ['CHEMBL4718_Ki', 'CHEMBL308_Ki', 'CHEMBL2147_Ki']

    SPECS_PATH = "data/screening_libraries/specs_cleaned.csv"
    ASINEX_PATH = "data/screening_libraries/asinex_cleaned.csv"
    ENAMINE_HIT_LOCATOR_PATH = "data/screening_libraries/enamine_hit_locator_cleaned.csv"

    # Load libraries
    library_specs = MoleculeDataset(pd.read_csv(SPECS_PATH)['smiles_cleaned'].tolist(),
                                    descriptor='ecfp', randomize_smiles=False)

    library_asinex = MoleculeDataset(pd.read_csv(ASINEX_PATH)['smiles_cleaned'].tolist(),
                                     descriptor='ecfp', randomize_smiles=False)

    library_enamine_hit_locator = MoleculeDataset(pd.read_csv(ENAMINE_HIT_LOCATOR_PATH)['smiles_cleaned'].tolist(),
                                                  descriptor='ecfp', randomize_smiles=False)

    libraries = {'asinex': library_asinex,
                 'enamine_hit_locator': library_enamine_hit_locator,
                 'specs': library_specs}

    # move to root dir
    os.chdir(ROOTDIR)

    for dataset_name in tqdm(datasets):
        print(dataset_name)

        best_hypers = rf_hyperparam_tuning(dataset_name, DEFAULT_SETTINGS_PATH, HYPERPARAM_GRID)

        settings = load_settings(DEFAULT_SETTINGS_PATH)
        config_dict = settings['training_config'] | {'dataset_name': dataset_name, 'out_path': ospj(settings['training_config']['out_path'] , dataset_name)}
        hyperparameters = settings['hyperparameters'] | best_hypers

        config = Config(**config_dict)
        config.set_hyperparameters(**hyperparameters)

        # make output dir
        prep_outdir(config)

        # save best hypers
        save_settings(config, ospj(config.out_path, "experiment_settings.yml"))

        # perform model training with cross validation and save results
        rf_cross_validate_and_inference(config, libraries)
