
import pandas as pd
import os

import torch

from jcm.datasets import MoleculeDataset
from jcm.trainer import train_lstm_vae, train_lstm_decoder
from jcm.config import Config, load_settings
# from constants import VAE_PRETRAIN_HYPERPARAMETERS
from constants import ROOTDIR
from dataprep.descriptors import one_hot_encode
from dataprep.complexity import split_smiles_by_complexity
OUT_DIR = os.path.join(ROOTDIR, 'results/chembl_vae_scaffs_trimmed')


settings = load_settings("experiments/hyperparams/vae_pretrain_0.yml")

experiment_name = settings['experiment_name']
hyperparameters = settings['hyperparameters']
training_config = settings['training_config']


if __name__ == '__main__':

    train_smiles = pd.read_csv('data/ChEMBL/chembl_train_smiles.csv').smiles.tolist()[:100000]
    complexity_splits = split_smiles_by_complexity(train_smiles, levels=training_config['curriculum_learning_splits'],
                                                   precomputed="data/ChEMBL/chembl_train_smiles_complexity.pt")

    model_checkpoint = None
    for complexity_level, split_idx in enumerate(complexity_splits):
        print(f"Complexity level {complexity_level}")

        split_smiles = [train_smiles[i] for i in split_idx]
        train_dataset = MoleculeDataset(split_smiles, descriptor=training_config['descriptor'])

        val_smiles = pd.read_csv('data/ChEMBL/chembl_val_smiles.csv').smiles.tolist()
        val_dataset = MoleculeDataset(val_smiles, descriptor=training_config['descriptor'])

        config = Config(**settings['training_config'])
        config.set_hyperparameters(**hyperparameters)

        # model, trainer = train_lstm_vae(config, train_dataset, val_dataset)
        model, trainer = train_lstm_vae(config, train_dataset, val_dataset, pre_trained_path=model_checkpoint)

        model_checkpoint = f"/Users/derekvantilborg/Dropbox/PycharmProjects/JointChemicalModel/results/chembl_vae_scaffs_trimmed/{experiment_name}_model_checkpoint_complexity_{complexity_level}.pt"
        torch.save(model.state_dict(), model_checkpoint)
