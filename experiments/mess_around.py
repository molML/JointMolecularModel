

import os
from os.path import join as ospj
import random
import pandas as pd
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from jcm.datasets import MoleculeDataset
from cheminformatics.encoding import encoding_to_smiles
import torch
from torch import nn as nn
from torch.nn import functional as F
from jcm.utils import get_smiles_length_batch
from jcm.models import DeNovoRNN, VAE, JointChemicalModel
from jcm.modules.rnn import AutoregressiveRNN, init_rnn_hidden, init_start_tokens
from cheminformatics.encoding import strip_smiles, probs_to_smiles
from cheminformatics.eval import smiles_validity, reconstruction_edit_distance
from constants import VOCAB
from collections import Counter
from jcm.config import Config, load_settings


data_path = ospj('data/split/ChEMBL_33_split.csv')
chembl = pd.read_csv(data_path)

train_smiles = chembl[chembl['split'] == 'train'].smiles.tolist()[:10000]
train_dataset = MoleculeDataset(train_smiles, descriptor='smiles', randomize_smiles=False)

val_smiles = chembl[chembl['split'] == 'val'].smiles.tolist()[:1000]
val_dataset = MoleculeDataset(val_smiles, descriptor='smiles', randomize_smiles=False)


# setup the dataloader
train_loader = DataLoader(train_dataset, batch_size=8)


from jcm.config import Config, load_settings
from jcm.training import Trainer
from jcm.callbacks import denovo_rnn_callback

experiment_settings = load_settings("experiments/hyperparams/jvae_default.yml")
experiment_settings['training_config']['batch_end_callback_every'] = 1000
experiment_settings['training_config']['save_every'] = 1000

config = Config(**experiment_settings['training_config'])
config.set_hyperparameters(**experiment_settings['hyperparameters'])



JointChemicalModel















# lstm_training_histories

# experiments = [i for i in os.listdir('results/rnn_pretraining') if not i.startswith('.')]
#
# all_results = []
# for exp_nr in experiments:
#     df = pd.read_csv(f'results/rnn_pretraining/{exp_nr}/training_history.csv')
#
#     # add best val loss + corresponding train loss and validity
#     results = {'experiment': exp_nr} | df.loc[df['val_loss'].idxmin()].to_dict()
#
#     config = load_settings(f'results/rnn_pretraining/{exp_nr}/experiment_settings.yml')
#     results = results | config['hyperparameters'] | config['training_config']
#
#     all_results.append(results)
#
# lstm_hyper_results = pd.DataFrame(all_results)
# lstm_hyper_results.to_csv('lstm_hyper_results.csv', index=False)
#
#
#
# experiments = [i for i in os.listdir('results/vae_pretraining') if not i.startswith('.')]
#
# all_results = []
# for exp_nr in experiments:
#     df = pd.read_csv(f'results/vae_pretraining/{exp_nr}/training_history.csv')
#
#     # add best val loss + corresponding train loss and validity
#     results = {'experiment': exp_nr} | df.loc[df['val_loss'].idxmin()].to_dict()
#
#     config = load_settings(f'results/vae_pretraining/{exp_nr}/experiment_settings.yml')
#     results = results | config['hyperparameters'] | config['training_config']
#
#     all_results.append(results)
#
# lstm_hyper_results = pd.DataFrame(all_results)
# lstm_hyper_results.to_csv('vae_hyper_results.csv', index=False)
#






data_path = ospj('data/split/ChEMBL_33_split.csv')
chembl = pd.read_csv(data_path)

train_smiles = chembl[chembl['split'] == 'train'].smiles.tolist()[:10000]
train_dataset = MoleculeDataset(train_smiles, descriptor='smiles', randomize_smiles=False)

val_smiles = chembl[chembl['split'] == 'val'].smiles.tolist()[:1000]
val_dataset = MoleculeDataset(val_smiles, descriptor='smiles', randomize_smiles=False)


# setup the dataloader
train_loader = DataLoader(train_dataset, batch_size=8)



from jcm.config import Config, load_settings
from jcm.training import Trainer
from jcm.callbacks import denovo_rnn_callback

experiment_settings = load_settings("experiments/hyperparams/vae_pretrain_default.yml")
experiment_settings['training_config']['batch_end_callback_every'] = 1000
experiment_settings['training_config']['save_every'] = 1000

experiment_settings['hyperparameters']['lr'] = 0.0003
experiment_settings['hyperparameters']['cnn_out_hidden'] = 256
experiment_settings['hyperparameters']['cnn_kernel_size'] = 6
experiment_settings['hyperparameters']['cnn_stride'] = 1
experiment_settings['hyperparameters']['cnn_n_layers'] = 2
experiment_settings['hyperparameters']['variational_scale'] = 0.1
experiment_settings['hyperparameters']['beta'] = 0.001
experiment_settings['hyperparameters']['z_size'] = 256
experiment_settings['hyperparameters']['rnn_type'] = 'gru'
experiment_settings['hyperparameters']['rnn_hidden_size'] = 512
experiment_settings['hyperparameters']['rnn_num_layers'] = 3

config = Config(**experiment_settings['training_config'])
config.set_hyperparameters(**experiment_settings['hyperparameters'])


model = VAE(config)
model.load_state_dict(torch.load("results/vae_pretraining/default/checkpoint_127000.pt", map_location=torch.device('cpu')))

# z = torch.rand(1000, 256)

z_dataset = MoleculeDataset(train_smiles[:100], descriptor='smiles', randomize_smiles=False)
z, original_smiles = model.get_z(z_dataset)

token_probs = model.generate(z, batch_size=256)

designs = probs_to_smiles(token_probs)

designs_clean = strip_smiles(designs)
validity, valid_smiles = smiles_validity(designs_clean, return_invalids=True)

print([smi for smi in valid_smiles if smi is not None])

for i, j in zip(original_smiles, valid_smiles):
    print(i == j, i, " > ", j)




random_factor = torch.rand(1, 256)

z = z + random_factor

token_probs = model.rnn.generate_from_z(z, 101)



