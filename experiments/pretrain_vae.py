
import pandas as pd
import os
from jcm.datasets import MoleculeDataset
from jcm.trainer import train_vae
from jcm.config import Config
from constants import VAE_PRETRAIN_HYPERPARAMETERS
from constants import ROOTDIR

OUT_DIR = os.path.join(ROOTDIR, 'results/chembl_vae')


if __name__ == '__main__':

    train_smiles = pd.read_csv('data/ChEMBL/train_smiles.csv').smiles.tolist()
    train_dataset = MoleculeDataset(train_smiles, descriptor='ecfp')

    val_smiles = pd.read_csv('data/ChEMBL/val_smiles.csv').smiles.tolist()
    val_dataset = MoleculeDataset(val_smiles, descriptor='ecfp')

    config = Config(max_iters=10000, batch_size=128, batch_end_callback_every=150, out_path=OUT_DIR)
    config.set_hyperparameters(**VAE_PRETRAIN_HYPERPARAMETERS)

    model = train_vae(config, train_dataset, val_dataset)
