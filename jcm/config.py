
import torch
import yaml
import numpy as np


# Helper function to convert numpy objects to serializable types
def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert array to list
    elif isinstance(obj, np.dtype):
        return str(obj)  # Convert dtype to string
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy(i) for i in obj)
    else:
        return obj


def load_settings(filename: str):
    with open(filename, 'r') as file:
        settings = yaml.safe_load(file)

    return settings


def save_settings(config, path: str = None):

    config_dict = {'training_config': convert_numpy(config.settings),
                   'hyperparameters': convert_numpy(config.hyperparameters)}

    with open(path, 'w') as file:
        yaml.dump(config_dict, file)


class Config:

    default_config = {'num_workers': 1, 'out_path': None}

    hyperparameters = {'lr': 3e-4}

    def __init__(self, **kwargs):
        self.merge_from_dict(self.default_config)
        self.merge_from_dict(kwargs)
        self.settings = self.default_config | kwargs

    def set_hyperparameters(self, **kwargs):

        if 'device' in kwargs:
            if kwargs['device'] == 'auto':
                kwargs['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.hyperparameters.update(kwargs)
        self.merge_from_dict(self.hyperparameters)

    def merge_from_dict(self, d):
        self.__dict__.update(d)

    def __repr__(self):
        vals = {k: v for k, v in self.__dict__.items() if k != 'hyperparameters'}
        return str(vals).replace(', ', '\n').replace('{', '').replace('}', '')
