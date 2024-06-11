

import os
from os.path import join as ospj
import random
import pandas as pd
import torch
from torch import Tensor
from torch import nn as nn
import torch.nn.functional as F
from sklearn.model_selection import ParameterGrid
from jcm.datasets import MoleculeDataset
from jcm.trainer import Trainer
from jcm.config import Config, load_settings
from cheminformatics.descriptors import one_hot_encode


PATH_TRAIN_SMILES = ospj('data', 'ChEMBL', 'chembl_train_smiles.csv')
PATH_VAL_SMILES = ospj('data', 'ChEMBL', 'chembl_val_smiles.csv')
DEFAULT_SETTINGS_PATH = "experiments/hyperparams/vae_pretrain_default.yml"

experiment_settings = load_settings(DEFAULT_SETTINGS_PATH)
config = experiment_settings['training_config']
config['batch_end_callback_every'] = 10
hypers = experiment_settings['hyperparameters']

train_smiles = pd.read_csv(PATH_TRAIN_SMILES).smiles.tolist()[:10000]
train_dataset = MoleculeDataset(train_smiles, descriptor=config['descriptor'], randomize_smiles=True)

val_smiles = pd.read_csv(PATH_VAL_SMILES).smiles.tolist()[:1000]
val_dataset = MoleculeDataset(val_smiles, descriptor=config['descriptor'], randomize_smiles=True)

config_ = Config(**config)
config_.set_hyperparameters(**hypers)


from jcm.models import DeNovoLSTM
from jcm.callbacks import denovo_lstm_callback

model = DeNovoLSTM(config_)

T = Trainer(config_, model, train_dataset, val_dataset)
if val_dataset is not None:
    T.set_callback('on_batch_end', denovo_lstm_callback)
T.run()








