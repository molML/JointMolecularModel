""" Perform hyperparameter tuning and model training for a MLP control model

Derek van Tilborg
Eindhoven University of Technology
March 2025
"""

import os
import copy
import torch
import pandas as pd
import numpy as np
from collections import defaultdict
from os.path import join as ospj
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, train_test_split, ParameterGrid
from sklearn.metrics import balanced_accuracy_score
from jcm.training_logistics import prep_outdir, train_model
from jcm.datasets import MoleculeDataset
from jcm.callbacks import mlp_callback
from jcm.config import Config, load_settings, save_settings
from jcm.utils import logits_to_pred
from constants import ROOTDIR
from jcm.models import SmilesMLP
from jcm.training import Trainer


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


def nn_mc_grid_search(model, callback, hyperparam_grid: dict[str, list], config: Config):

    history = defaultdict(list)
    for hypers in ParameterGrid(hyperparam_grid):

        config_ = copy.copy(config)
        updated_hypers = config_.hyperparameters | hypers
        config_.set_hyperparameters(**updated_hypers)

        n = config_.n_cross_validate_hyperparam
        seeds = np.random.default_rng(seed=config_.random_state).integers(0, 1000, n)

        all_val_losses = []
        for seed in seeds:

            # take a fold from the dataset
            train_dataset, val_dataset = load_full_data(config_, val_size=config_.val_size, random_state=seed)
            # train a model
            _model, trainer = train_model(model, callback, config_, train_dataset, val_dataset)
            # add the lowest validation loss to the list of all_val_losses
            all_val_losses.append(min(trainer.history['val_loss']))

        # take the mean over the n folds and add it to the history
        history['val_loss'].append(sum(all_val_losses)/len(all_val_losses))
        history['hypers'].append(hypers)

    return history


def mlp_mc_hyperparam_tuning(model, callback, dataset_name: str, default_config_path: str, hyper_grid: dict[str, list]) -> dict:
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
    grid_search_history = nn_mc_grid_search(model, callback, hyper_grid, config)

    # Fit the grid search to the data
    print("Starting hyperparameter tuning")
    best_hypers = grid_search_history['hypers'][np.argmin(grid_search_history['val_loss'])]

    # Print the best parameters
    print("Best parameters found: ", best_hypers)
    print("Best cross-validation score: ", min(grid_search_history['val_loss']))

    return best_hypers


def nn_mc_cross_validate_and_inference(model, callback, config: Config):
    """

    :param config:
    :return:
    """

    n = config.n_cross_validate
    master_seed = config.random_state
    seeds = np.random.default_rng(seed=master_seed).integers(0, 1000, n)
    out_path = ospj(config.out_path)
    config.out_path = None

    results = []
    metrics = []
    for seed in seeds:
        # split a chunk of the train data, we don't use the validation data in the RF approach, but we perform cross-
        # validation using the same strategy so we can directly compare methods.
        try:
            train_dataset, val_dataset = load_full_data(config, val_size=config.val_size, random_state=seed)

            # train model and pickle it afterwards
            _model, trainer = train_model(model, callback, config, train_dataset, val_dataset)

            torch.save(_model, ospj(out_path, f"model_{seed}.pt"))

            # perform predictions on all splits
            results_train = _model.predict(train_dataset)
            results_val = _model.predict(val_dataset)

            logits_N_K_C_train, y_train = results_train["y_logprobs_N_K_C"], results_train["y"]
            logits_N_K_C_val, y_val = results_val["y_logprobs_N_K_C"], results_val["y"]

            y_hat_train, y_unc_train = logits_to_pred(logits_N_K_C_train, return_binary=True)
            y_hat_val, y_unc_val = logits_to_pred(logits_N_K_C_val, return_binary=True)

            # Put the predictions in a dataframe
            train_results_df = pd.DataFrame({'seed': seed, 'split': 'train', 'smiles': train_dataset.smiles,
                                             'y': y_train.cpu(), 'y_hat': y_hat_train.cpu(), 'y_unc': y_unc_train.cpu()})
            val_results_df = pd.DataFrame({'seed': seed, 'split': 'val', 'smiles': val_dataset.smiles,
                                            'y': y_val.cpu(), 'y_hat': y_hat_val.cpu(), 'y_unc': y_unc_val.cpu()})

            all_dfs = [train_results_df, val_results_df]
            if libraries is not None:
                for library_name, library in libraries.items():
                    print(f"performing inference on {library_name} ({seed})")

                    results_lib= _model.predict(train_dataset)

                    # perform predictions on all splits
                    logits_N_K_C_lib = results_train["y_logprobs_N_K_C"]

                    y_hat_lib, y_unc_lib = logits_to_pred(logits_N_K_C_lib, return_binary=True)

                    lib_results_df = pd.DataFrame({'seed': seed, 'split': library_name, 'smiles': library.smiles,
                                                   'y': None, 'y_hat': y_hat_lib, 'y_unc': y_unc_lib})

                    all_dfs.append(lib_results_df)

            results_df = pd.concat(all_dfs)
            results.append(results_df)

            # Put the performance metrics in a dataframe
            metrics.append({'seed': seed,
                            'train_balanced_acc': balanced_accuracy_score(train_dataset.y.cpu(), y_hat_train.cpu()),
                            'train_mean_uncertainty': torch.mean(y_unc_train).item(),
                            'val_balanced_acc': balanced_accuracy_score(val_dataset.y.cpu(), y_hat_val.cpu()),
                            'val_mean_uncertainty': torch.mean(y_unc_val).item()
                            })

            # log the results/metrics
            pd.concat(results).to_csv(ospj(out_path, 'results_preds.csv'), index=False)
            pd.DataFrame(metrics).to_csv(ospj(out_path, 'results_metrics.csv'), index=False)

        except:
            print(f"Failed seed {seed}. Skipping this one")


if __name__ == '__main__':

    MODEL = SmilesMLP
    CALLBACK = mlp_callback
    EXPERIMENT_NAME = "smiles_mlp_prospective"
    DEFAULT_SETTINGS_PATH = "experiments/hyperparams/smiles_mlp_default_prospective.yml"
    HYPERPARAM_GRID = {'mlp_hidden_dim': [1024, 2048],
                       'mlp_n_layers': [2, 3, 4, 5],
                       'lr': [3e-4, 3e-5, 3e-6],
                       'data_augmentation': [False],
                       'cnn_out_hidden': [256, 512],
                       'cnn_kernel_size': [6, 8],
                       'cnn_n_layers': [2, 3],
                       'cnn_dropout': [0, 0.1],
                       'z_size': [128],
                       'weight_decay': [0, 0.0001]
                       }


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
        # break
        print(dataset_name)

        best_hypers = mlp_mc_hyperparam_tuning(MODEL, CALLBACK, dataset_name, DEFAULT_SETTINGS_PATH, HYPERPARAM_GRID)

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
        nn_mc_cross_validate_and_inference(config, libraries)
