
import pandas as pd
import os
from jcm.datasets import MoleculeDataset
from jcm.trainer import train_vae
from jcm.config import Config
from constants import VAE_PRETRAIN_HYPERPARAMETERS
from constants import ROOTDIR
OUT_DIR = os.path.join(ROOTDIR, 'results/chembl_vae')


#### Hyperparameters space
# Beta: 1e-2, 1e-3, 1e-4
# lr:   3e-3, 3e-4, 3e-5
# variational_scale: 0.1, 0.01, 0.001

if __name__ == '__main__':

    train_smiles = pd.read_csv('data/ChEMBL/train_smiles.csv').smiles.tolist()
    train_dataset = MoleculeDataset(train_smiles, descriptor='ecfp')

    val_smiles = pd.read_csv('data/ChEMBL/val_smiles.csv').smiles.tolist()
    val_dataset = MoleculeDataset(val_smiles, descriptor='ecfp')

    VAE_PRETRAIN_HYPERPARAMETERS['latent_dim'] = 64

    config = Config(max_iters=200000, batch_size=256, batch_end_callback_every=1000, out_path=OUT_DIR)
    config.set_hyperparameters(**VAE_PRETRAIN_HYPERPARAMETERS)

    model, trainer = train_vae(config, train_dataset, val_dataset)
