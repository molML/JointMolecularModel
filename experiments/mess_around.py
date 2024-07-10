

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
from jcm.training import Trainer
from jcm.config import Config, load_settings



data_path = ospj('data/split/ChEMBL_33_split.csv')
chembl = pd.read_csv(data_path)

# DEFAULT_SETTINGS_PATH = "experiments/hyperparams/vae_pretrain_default.yml"
DEFAULT_SETTINGS_PATH = "experiments/hyperparams/autoregressive_lstm_default.yml"


experiment_settings = load_settings(DEFAULT_SETTINGS_PATH)
config = experiment_settings['training_config']
config['batch_end_callback_every'] = 10
hypers = experiment_settings['hyperparameters']

train_smiles = chembl[chembl['split'] == 'train'].smiles.tolist()[:1000000]
train_dataset = MoleculeDataset(train_smiles, descriptor=config['descriptor'], randomize_smiles=False)

val_smiles = chembl[chembl['split'] == 'val'].smiles.tolist()[:10000]
val_dataset = MoleculeDataset(val_smiles, descriptor=config['descriptor'], randomize_smiles=False)

config_ = Config(**config)
config_.set_hyperparameters(**hypers)


from jcm.models import SmilesMLP, ECFPMLP, JointChemicalModel, VAE, DeNovoLSTM
from jcm.callbacks import mlp_callback, jvae_callback, vae_callback, denovo_lstm_callback


model = DeNovoLSTM(config_)


T = Trainer(config_, model, train_dataset, val_dataset)
if val_dataset is not None:
    T.set_callback('on_batch_end', denovo_lstm_callback)
T.run()

