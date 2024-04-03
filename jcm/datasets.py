
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from dataprep.utils import smiles_to_mols
from dataprep.descriptors import mols_to_ecfp, mols_to_maccs, encode_smiles
from jcm.utils import to_binary


class MoleculeDataset(Dataset):

    allowed_descriptors = ['ecfp', 'maccs', 'smiles']

    def __init__(self, smiles: list[str], y=None, descriptor: str = 'ecfp', descriptor_kwargs=None, **kwargs):
        if descriptor_kwargs is None:
            descriptor_kwargs = {}
        self.smiles = smiles
        self.y = y
        self.descriptor = descriptor
        self.descriptor_kwargs = descriptor_kwargs
        self.__dict__.update(kwargs)

        if y is not None:
            self.y = torch.tensor([self.y]) if type(self.y) is not torch.Tensor else self.y
            assert len(smiles) == len(y), 'The number of labels must match the number of molecules'
        assert descriptor in self.allowed_descriptors, f'the descriptor must be: {self.allowed_descriptors}'

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):

        if type(idx) is int:
            idx = [idx]

        smiles = [self.smiles[i] for i in list(idx)]

        if self.descriptor == 'ecfp':
            mols = smiles_to_mols(smiles)
            x = mols_to_ecfp(mols, to_array=True, **self.descriptor_kwargs)
            x = torch.tensor(x)
        elif self.descriptor == 'maccs':
            mols = smiles_to_mols(smiles)
            x = mols_to_maccs(mols, to_array=True, **self.descriptor_kwargs)
            x = torch.tensor(x)
        elif self.descriptor == 'smiles':
            x = encode_smiles(smiles)

        if self.y is not None:
            y = self.y[idx]
            return x, y
        return x


def load_moleculeace(filename: str, val_split: float = 0.2, seed: int = 42,
                     classification_threshold: float = 100, descriptor_kwargs: dict = None,
                     descriptor: str = 'ecfp') -> (Dataset, Dataset, Dataset):
    """ Load MoleculeACE datasets into a train/val/test dataset

    :param filename: path of the csv
    :param val_split: ratio of data to split from the train set (default=0.2)
    :param seed: random seed determining the validation split
    :param classification_threshold: threshold to determine classes (default=100nM)
    :param descriptor: 'ecfp' or 'maccs' (default='ecfp')
    :descriptor_kwargs: dict containing kwargs for the descriptors (default=None)
    :return: train_dataset, val_dataset, test_dataset
    """
    df = pd.read_csv(filename)
    df.y = to_binary(torch.tensor(df['exp_mean [nM]'].tolist()), threshold=classification_threshold)
    df_train = df[df['ood_split'] == 'train'].reset_index()

    # get a random validation split from the train data
    rng = np.random.default_rng(seed)
    val_idx = rng.choice(range(len(df_train)), int(len(df_train)*val_split), replace=False)
    train_idx = np.array([i for i in range(len(df_train)) if i not in val_idx])

    df_val = df_train.iloc[val_idx, :]
    df_train = df_train.iloc[train_idx, :]
    df_test = df[df['ood_split'] == 'test'].reset_index()

    train_dataset = MoleculeDataset(df_train.smiles.tolist(), torch.tensor(df_train.y.tolist()),
                                    sim_to_train_medoid=df_train.sim_to_train_medoid.tolist(),
                                    descriptor=descriptor, descriptor_kwargs=descriptor_kwargs)

    val_dataset = MoleculeDataset(df_val.smiles.tolist(), torch.tensor(df_val.y.tolist()),
                                  sim_to_train_medoid=df_val.sim_to_train_medoid.tolist(),
                                  descriptor=descriptor, descriptor_kwargs=descriptor_kwargs)

    test_dataset = MoleculeDataset(df_test.smiles.tolist(), torch.tensor(df_test.y.tolist()),
                                   sim_to_train_medoid=df_test.sim_to_train_medoid.tolist(),
                                   descriptor=descriptor, descriptor_kwargs=descriptor_kwargs)

    return train_dataset, val_dataset, test_dataset
