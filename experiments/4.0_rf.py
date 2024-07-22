

import os
from os.path import join as ospj
from itertools import batched
import pandas as pd
from sklearn.model_selection import ParameterGrid
from jcm.callbacks import vae_callback
from jcm.config import Config, load_settings
from jcm.datasets import MoleculeDataset
from jcm.models import VAE
from jcm.training import Trainer
from constants import ROOTDIR
import argparse
from sklearn.model_selection import train_test_split


# 1. dataset
# 2. split into train/val
# 3. perform hyperparam tuning
# 4. bioactivity prediction on datasets
# 5. save to file


def load_datasets(config):

    data_path = ospj(f'data/split/{config.dataset_name}_split.csv')

    # get the train and val SMILES from the pre-processed file
    data = pd.read_csv(data_path)

    train_data = data[data['split'] == 'train']
    test_data = data[data['split'] == 'test']
    ood_data = data[data['split'] == 'ood']

    # Perform a random split of the train data into train and validation
    train_data, val_data = train_test_split(train_data, test_size=config.val_size, random_state=config.random_state)

    # Initiate the datasets
    train_dataset = MoleculeDataset(train_data.smiles.tolist(), train_data.y.tolist(),
                                    descriptor=config.descriptor, randomize_smiles=config.data_augmentation)

    val_dataset = MoleculeDataset(val_data.smiles.tolist(),  val_data.y.tolist(),
                                  descriptor=config.descriptor,  randomize_smiles=config.data_augmentation)

    test_dataset = MoleculeDataset(test_data.smiles.tolist(), test_data.y.tolist(),
                                   descriptor=config.descriptor, randomize_smiles=config.data_augmentation)

    ood_dataset = MoleculeDataset(ood_data.smiles.tolist(), ood_data.y.tolist(),
                                  descriptor=config.descriptor, randomize_smiles=config.data_augmentation)

    return train_dataset, val_dataset, test_dataset, ood_dataset


def hyperparam_tuning():
    pass


def train_model(config):
    pass


def evaluate_model():
    pass


def train_n_models(n: int = 10):

    for seed in range(n):
        pass
        # train model

    pass


if __name__ == '__main__':
    pass

    hyperparameters = {'n_estimators': [100, 250, 500, 1000],
                       'max_depth': [None, 10, 20, 30],
                       'min_samples_split': [2, 5, 10]}


    DEFAULT_SETTINGS_PATH = "experiments/hyperparams/rf_default.yml"

    experiment_settings = load_settings(DEFAULT_SETTINGS_PATH)
    default_config_dict = experiment_settings['training_config']
    default_config_dict['dataset_name'] = 'CHEMBL233_Ki'
    default_hyperparameters = experiment_settings['hyperparameters']

    config = Config(**default_config_dict)
    config.set_hyperparameters(**default_hyperparameters)

    train_dataset, val_dataset, test_dataset, ood_dataset = load_datasets(config)



