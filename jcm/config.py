
import sys
import os
import torch
import yaml
import numpy as np
import wandb
from constants import WANDB_KEY


# Helper function to convert numpy objects to serializable types
def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        if obj.shape == ():
            return obj.item()
        return obj.tolist()  # Convert array to list
    elif isinstance(obj, np.dtype):
        return str(obj)  # Convert dtype to string
    elif isinstance(obj, np.generic):
        return obj.item()  # Convert NumPy scalar to Python scalar
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

    print(config.settings)
    print(config.hyperparameters)

    with open(path, 'w') as file:
        yaml.safe_dump(config_dict, file, default_flow_style=False)


def load_and_setup_config_from_file(path: str | dict, config_dict: dict = None, hyperparameters: dict = None):
    """ if path is a dict, use that, else load dict from the path. Will be updated according to the condig_dict and
     hyperparameters """

    if type(path) is dict:
        settings = path
    else:
        settings = load_settings(path)

    if config_dict:
        config_dict = settings['training_config'] | config_dict
    else:
        config_dict = settings['training_config']

    if hyperparameters:
        hyperparameters = settings['hyperparameters'] | hyperparameters
    else:
        hyperparameters = settings['hyperparameters']

    config = Config(**config_dict)
    config.set_hyperparameters(**hyperparameters)

    return config


def init_experiment(config_path: str | dict, config_dict: dict = None, hyperparameters: dict = None,
                    name: str = None, group: str = None, job_type: str = None,
                    project: str = "JointChemicalModel", **kwargs):

    if wandb.run is None:
        os.environ["WANDB_API_KEY"] = WANDB_KEY
    else:
        print("Another WandB session is already running. Re-initiating session.", file=sys.stderr)
        finish_experiment()

    config = load_and_setup_config_from_file(config_path, config_dict=config_dict, hyperparameters=hyperparameters)

    wandb.init(
        job_type=job_type,
        group=group,
        project=project,
        name=name,
        config=config.get_everything(),
        reinit=True,
        **kwargs
    )

    return config


def finish_experiment():
    if wandb.run is not None:
        wandb.finish()


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

    def get_everything(self):
        return self.hyperparameters | self.settings

    def __repr__(self):
        vals = {k: v for k, v in self.__dict__.items() if k != 'hyperparameters'}
        return str(vals).replace(', ', '\n').replace('{', '').replace('}', '')
