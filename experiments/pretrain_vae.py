
import pandas as pd
import os
from os.path import join as ospj
import torch
from jcm.datasets import MoleculeDataset
from jcm.trainer import train_lstm_vae, train_lstm_decoder
from jcm.config import Config, load_settings
from constants import ROOTDIR
from dataprep.descriptors import one_hot_encode
from dataprep.complexity import split_smiles_by_complexity



SETTINGS = load_settings("experiments/hyperparams/vae_pretrain_0.yml")

HYPERS = SETTINGS['hyperparameters']
CONFIG = SETTINGS['training_config']

PATH_TRAIN_SMILES = ospj('data', 'ChEMBL', 'chembl_train_smiles.csv')
PATH_VAL_SMILES = ospj('data', 'ChEMBL', 'chembl_val_smiles.csv')
PATH_TRAIN_SMILES_COMPLEXITY = ospj("data", "ChEMBL", "chembl_train_smiles_complexity.pt")
PATH_OUT = ospj(CONFIG['out_path'], CONFIG['experiment_name'])

# torch.cuda.is_available()


if __name__ == '__main__':

    os.makedirs(PATH_OUT, exist_ok=True)

    train_smiles = pd.read_csv(PATH_TRAIN_SMILES).smiles.tolist()[:100000]
    complexity_splits = split_smiles_by_complexity(train_smiles,
                                                   levels=CONFIG['curriculum_learning_splits'],
                                                   precomputed=PATH_TRAIN_SMILES_COMPLEXITY)

    val_smiles = pd.read_csv(PATH_VAL_SMILES).smiles.tolist()
    val_dataset = MoleculeDataset(val_smiles, descriptor=CONFIG['descriptor'])

    model_checkpoint = None
    for complexity_level, split_idx in enumerate(complexity_splits):
        print(f"Complexity level {complexity_level}")

        split_smiles = [train_smiles[i] for i in split_idx]
        train_dataset = MoleculeDataset(train_smiles, descriptor=CONFIG['descriptor'])

        config = Config(**CONFIG)
        config.set_hyperparameters(**HYPERS)

        # model, trainer = train_lstm_vae(config, train_dataset, val_dataset)
        model, trainer = train_lstm_vae(config, train_dataset, val_dataset, pre_trained_path=model_checkpoint)

        model_checkpoint = ospj(PATH_OUT, f"model_checkpoint_complexity_{complexity_level}.pt")
        torch.save(model.state_dict(), model_checkpoint)

