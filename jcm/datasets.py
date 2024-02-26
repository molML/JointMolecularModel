
import torch
from torch.utils.data import Dataset
from dataprep.utils import smiles_to_mols
from dataprep.descriptors import mols_to_ecfp, mols_to_maccs


class MoleculeDataset(Dataset):

    allowed_descriptors = ['ecfp', 'maccs']
    def __init__(self, smiles: list[str], y=None, descriptor: str = 'ecfp', **kwargs):
        self.smiles = smiles
        self.targets = y
        self.descriptor = descriptor
        self.kwargs = kwargs

        if y is not None:
            self.targets = torch.tensor([self.targets]) if type(self.targets) is not torch.Tensor else self.targets
            assert len(smiles) == len(y), 'The number of labels must match the number of molecules'
        assert descriptor in self.allowed_descriptors, f'the descriptor must be: {self.allowed_descriptors}'

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):

        if type(idx) is int:
            idx = [idx]

        smiles = [self.smiles[i] for i in list(idx)]
        mols = smiles_to_mols(smiles)

        if self.descriptor == 'ecfp':
            x = mols_to_ecfp(mols, to_array=True, **self.kwargs)
        elif self.descriptor == 'maccs':
            x = mols_to_maccs(mols, to_array=True, **self.kwargs)

        if self.targets is not None:
            y = self.targets[idx]
            return torch.tensor(x), y
        return torch.tensor(x)
