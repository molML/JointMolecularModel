

import os
from os.path import join as ospj
import pandas as pd
from jcm.datasets import MoleculeDataset
from jcm.models import DeNovoRNN, VAE
from jcm.config import Config, load_settings
from jcm.training import Trainer
from jcm.callbacks import denovo_rnn_callback, vae_callback


data_path = ospj('data/split/ChEMBL_33_split.csv')
chembl = pd.read_csv(data_path)

train_smiles = chembl[chembl['split'] == 'train'].smiles.tolist()  #[:10000]
train_dataset = MoleculeDataset(train_smiles, descriptor='smiles', randomize_smiles=False)

val_smiles = chembl[chembl['split'] == 'val'].smiles.tolist()  #[:1000]
val_dataset = MoleculeDataset(val_smiles, descriptor='smiles', randomize_smiles=False)

# experiment_settings = load_settings("experiments/hyperparams/autoregressive_rnn_default.yml")
experiment_settings = load_settings("experiments/hyperparams/vae_pretrain_default.yml")
experiment_settings['training_config']['batch_end_callback_every'] = 5
experiment_settings['training_config']['batch_size'] = 64
experiment_settings['training_config']['val_molecules_to_sample'] = 64
experiment_settings['training_config']['max_iters'] = 20

experiment_settings['hyperparameters']['lr'] = 0.0003
experiment_settings['hyperparameters']['cnn_out_hidden'] = 256
experiment_settings['hyperparameters']['cnn_kernel_size'] = 6
experiment_settings['hyperparameters']['cnn_stride'] = 1
experiment_settings['hyperparameters']['cnn_n_layers'] = 2
experiment_settings['hyperparameters']['variational_scale'] = 0.1
experiment_settings['hyperparameters']['beta'] = 0.001
experiment_settings['hyperparameters']['rnn_type'] = 'gru'
experiment_settings['hyperparameters']['z_size'] = 128
experiment_settings['hyperparameters']['rnn_hidden_size'] = 256
experiment_settings['hyperparameters']['rnn_num_layers'] = 1
experiment_settings['hyperparameters']['rnn_teacher_forcing'] = True


config = Config(**experiment_settings['training_config'])
config.set_hyperparameters(**experiment_settings['hyperparameters'])


# model = DeNovoRNN(config)
model = VAE(config)

T = Trainer(config, model, train_dataset, val_dataset)
if val_dataset is not None:
    T.set_callback('on_batch_end', vae_callback)
T.run()
