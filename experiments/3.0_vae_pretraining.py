""" Run a model for hyperparameter tuning. This script contains a function to write the specific SLURM scripts I use on
our computer cluster, that in turn run this script with a certain set of hyperparameters

Derek van Tilborg
Eindhoven University of Technology
June 2024
"""

import os
from os.path import join as ospj
import pandas as pd
from sklearn.model_selection import ParameterGrid
from jcm.callbacks import vae_callback
from jcm.config import Config, load_settings
from jcm.datasets import MoleculeDataset
from jcm.models import VAE
from jcm.training import Trainer
from constants import ROOTDIR
import argparse


def load_datasets():

    data_path = ospj('data/split/ChEMBL_33_split.csv')

    # get the train and val SMILES from the pre-processed file
    chembl = pd.read_csv(data_path)
    train_smiles = chembl[chembl['split'] == 'train'].smiles.tolist()
    val_smiles = chembl[chembl['split'] == 'val'].smiles.tolist()

    # Initiate the datasets
    train_dataset = MoleculeDataset(train_smiles, descriptor='smiles', randomize_smiles=True)
    val_dataset = MoleculeDataset(val_smiles, descriptor='smiles', randomize_smiles=True)

    return train_dataset, val_dataset


def configure_config(hypers: dict = None, settings: dict = None):

    DEFAULT_SETTINGS_PATH = "experiments/hyperparams/vae_pretrain_default.yml"

    experiment_settings = load_settings(DEFAULT_SETTINGS_PATH)
    default_config_dict = experiment_settings['training_config']
    default_hyperparameters = experiment_settings['hyperparameters']

    # update settings
    if settings is not None:
        default_config_dict = default_config_dict | settings

    # update hyperparameters
    if hypers is not None:
        default_hyperparameters = default_hyperparameters | hypers

    config = Config(**default_config_dict)
    config.set_hyperparameters(**default_hyperparameters)

    return config


def train_model(config):
    """ Train a model according to the config

    :param config: Config object containing all settings and hypers
    """
    train_dataset, val_dataset = load_datasets()

    model = VAE(config)

    T = Trainer(config, model, train_dataset, val_dataset)
    if val_dataset is not None:
        T.set_callback('on_batch_end', vae_callback)
    T.run()


def write_job_script(hyperparameters):
    pass


if __name__ == '__main__':

    # parse script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', help='The path of the output directory', default='results')
    parser.add_argument('-experiment')
    args = parser.parse_args()

    # move to root dir
    os.chdir(ROOTDIR)

    # global variables
    SEARCH_SPACE = {'lr': [3e-3, 3e-4, 3e-5],
                    'cnn_out_hidden': [256, 512],
                    'cnn_kernel_size': [6, 8],
                    'cnn_n_layers': [2, 3],
                    'z_size': [128],
                    'lstm_hidden_size': [256, 512],
                    'lstm_num_layers': [2, 3],
                    'lstm_dropout': [0.2],
                    'variational_scale': [0.1],
                    'beta': [0.001, 0.0001],
                   }

    out_path = [args.o]
    experiment = [args.experiment]

    # out_path = 'results/vae_pretraining'
    # experiment = 1

    experiment_hypers = ParameterGrid(SEARCH_SPACE)[experiment]
    experiment_settings = {'out_path': out_path, 'experiment_name': str(experiment)}

    config = configure_config(hypers=experiment_hypers, settings=experiment_settings)

    train_model(config)
