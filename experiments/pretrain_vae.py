
import pandas as pd
from jcm.datasets import MoleculeDataset
from jcm.trainer import train_vae
from jcm.config import Config

VAE_HYPERPARAMETERS = {'input_dim': 1024,
                       "latent_dim": 128,
                       'out_dim': 1024,
                       'beta': 0.001,
                       'class_scaling_factor': 20,
                       'lr': 3e-4}


if __name__ == '__main__':

    train_smiles = pd.read_csv('data/ChEMBL/train_smiles.csv').smiles.tolist()
    train_dataset = MoleculeDataset(train_smiles, descriptor='ecfp')

    val_smiles = pd.read_csv('data/ChEMBL/val_smiles.csv').smiles.tolist()
    val_dataset = MoleculeDataset(val_smiles, descriptor='ecfp')

    config = Config(max_iters=10000, batch_size=128, batch_end_callback_every=150, out_path=None)
    config.set_hyperparameters(**VAE_HYPERPARAMETERS)

    model = train_vae(config, train_dataset, val_dataset)
