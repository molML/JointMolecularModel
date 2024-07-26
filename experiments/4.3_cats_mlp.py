""" Perform hyperparameter tuning and model training for a cats + MLP control model

Derek van Tilborg
Eindhoven University of Technology
July 2024
"""

import os
from os.path import join as ospj
from tqdm import tqdm
from jcm.config import Config, load_settings, save_settings
from jcm.training_logistics import prep_outdir, get_all_datasets, nn_hyperparam_tuning, nn_cross_validate
from constants import ROOTDIR


if __name__ == '__main__':

    DEFAULT_SETTINGS_PATH = "experiments/hyperparams/cats_mlp_default.yml"
    HYPERPARAM_GRID = {'mlp_hidden_dim': [1024, 2048],
                       'mlp_n_layers': [2, 3, 4, 5],
                       'lr': [3e-4, 3e-5, 3e-6]}

    # move to root dir
    os.chdir(ROOTDIR)

    all_datasets = get_all_datasets()
    for dataset_name in tqdm(all_datasets):
        print(dataset_name)

        best_hypers = nn_hyperparam_tuning(dataset_name, DEFAULT_SETTINGS_PATH, HYPERPARAM_GRID)

        settings = load_settings(DEFAULT_SETTINGS_PATH)
        config_dict = settings['training_config'] | {'dataset_name': dataset_name}
        hyperparameters = settings['hyperparameters'] | best_hypers

        config = Config(**config_dict)
        config.set_hyperparameters(**hyperparameters)

        # make output dir
        prep_outdir(config)

        # save best hypers
        save_settings(config, ospj(config.out_path, config.experiment_name, config.dataset_name, 'experiment_settings.yml'))

        # perform model training with cross validation and save results
        results = nn_cross_validate(config)
