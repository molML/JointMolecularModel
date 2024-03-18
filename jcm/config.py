import torch
import os
from constants import DEFAULT_CONFIG


class Config:

    default_config = DEFAULT_CONFIG

    hyperparameters = {'lr': 3e-4}

    def __init__(self, **kwargs):
        self.merge_from_dict(self.default_config)
        self.merge_from_dict(kwargs)

    def set_hyperparameters(self, **kwargs):
        self.hyperparameters.update(kwargs)
        self.merge_from_dict(self.hyperparameters)

    def merge_from_dict(self, d):
        self.__dict__.update(d)

    def __repr__(self):
        vals = {k: v for k, v in self.__dict__.items() if k != 'hyperparameters'}
        return str(vals).replace(', ', '\n').replace('{', '').replace('}', '')
