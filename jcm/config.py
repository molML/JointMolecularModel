import torch
import os

ROOTDIR = "/Users/derekvantilborg/Dropbox/PycharmProjects/JointChemicalModel"


class Config:

    default_config = {'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                      'batch_size': 128,
                      'num_workers': 1,
                      'batch_end_callback_every': 1000,
                      'val_molecules_to_sample': 1000,
                      'out_path': os.path.join(ROOTDIR, 'results/chembl_vae')}

    hyperparameters = {'lr': 3e-4}

    def __init__(self, **kwargs):
        self.merge_from_dict(self.default_config)
        self.merge_from_dict(kwargs)

    def set_hyperparameters(self, **kwargs):
        self.hyperparameters = kwargs
        self.merge_from_dict(kwargs)

    def merge_from_dict(self, d):
        self.__dict__.update(d)

    def __repr__(self):
        return str(self.__dict__).replace(', ', '\n').replace('{', '').replace('}', '')
