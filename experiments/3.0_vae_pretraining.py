""" Run a model for hyperparameter tuning. This script contains a function to write the specific SLURM scripts I use on
our computer cluster, that in turn run this script with a certain set of hyperparameters

Derek van Tilborg
Eindhoven University of Technology
June 2024
"""

import os
from os.path import join as ospj
from itertools import batched
import pandas as pd
from sklearn.model_selection import ParameterGrid
from jcm.callbacks import vae_callback
from jcm.config import Config, load_settings
from jcm.datasets import MoleculeDataset
from jcm.models import VAE
from jcm.training import Trainer
from constants import ROOTDIR
import argparse


def load_datasets():

    data_path = ospj('data/split/ChEMBL_33_split.csv')

    # get the train and val SMILES from the pre-processed file
    chembl = pd.read_csv(data_path)
    train_smiles = chembl[chembl['split'] == 'train'].smiles.tolist()
    val_smiles = chembl[chembl['split'] == 'val'].smiles.tolist()

    # Initiate the datasets
    train_dataset = MoleculeDataset(train_smiles, descriptor='smiles', randomize_smiles=True)
    val_dataset = MoleculeDataset(val_smiles, descriptor='smiles', randomize_smiles=True)

    return train_dataset, val_dataset


def configure_config(hypers: dict = None, settings: dict = None):

    DEFAULT_SETTINGS_PATH = "experiments/hyperparams/vae_pretrain_default.yml"

    experiment_settings = load_settings(DEFAULT_SETTINGS_PATH)
    default_config_dict = experiment_settings['training_config']
    default_hyperparameters = experiment_settings['hyperparameters']

    # update settings
    if settings is not None:
        default_config_dict = default_config_dict | settings

    # update hyperparameters
    if hypers is not None:
        default_hyperparameters = default_hyperparameters | hypers

    config = Config(**default_config_dict)
    config.set_hyperparameters(**default_hyperparameters)

    return config


def train_model(config):
    """ Train a model according to the config

    :param config: Config object containing all settings and hypers
    """
    train_dataset, val_dataset = load_datasets()

    model = VAE(config)

    T = Trainer(config, model, train_dataset, val_dataset)
    if val_dataset is not None:
        T.set_callback('on_batch_end', vae_callback)
    T.run()


def write_job_script(experiments: list[int], experiment_name: str = "vae_pretraining",
                     experiment_script: str = "3.0_vae_pretraining.py", partition: str = 'gpu', ntasks: str = '18',
                     gpus_per_node: str = 1, time: str = "4:00:00") -> None:
    """
    :param experiments: list of experiment numbers, e.g. [0, 1, 2]
    """

    jobname = experiment_name + '_' + '_'.join([str(i) for i in experiments])

    lines = []
    lines.append('# !/bin/bash\n')
    lines.append(f'# SBATCH --job-name={jobname}\n')
    lines.append(f'# SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/{jobname}.out\n')
    lines.append(f'# SBATCH -p {partition}\n')
    lines.append('# SBATCH -N 1\n')
    lines.append(f'# SBATCH --ntasks={ntasks}\n')
    lines.append(f'# SBATCH --gpus-per-node={gpus_per_node}\n')
    lines.append(f'# SBATCH --time={time}\n')
    lines.append('\n')
    lines.append(f'experiment_name="{experiment_name}"\n')
    lines.append('\n')
    lines.append('project_path="$HOME/projects/JointChemicalModel"\n')
    lines.append(f'experiment_script_path="$project_path/experiments/{experiment_script}"\n')
    lines.append('\n')
    lines.append('out_path="$project_path/results/$experiment_name"\n')
    lines.append('log_path="$project_path/results/logs"\n')
    lines.append('\n')
    lines.append('source $HOME/anaconda3/etc/profile.d/conda.sh\n')
    lines.append('export PYTHONPATH="$PYTHONPATH:$project_path"\n')

    for i, exp in enumerate(experiments):
        lines.append('\n')
        lines.append('$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o $out_path -experiment EX > "$log_path/${experiment_name}_EX.log" &\n'.replace('EX', str(exp)))
        lines.append(f'pid{i+1}=$!\n')

    lines.append('\n')
    for i, exp in enumerate(experiments):
        lines.append(f'wait $pid{i+1}\n')
    lines.append('\n')

    outpath = ospj(ROOTDIR, 'experiments', 'jobs', jobname + '.sh')

    # Write the modified lines back to the file
    with open(outpath, 'w') as file:
        file.writelines(lines)


if __name__ == '__main__':

    # parse script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', help='The path of the output directory', default='results')
    parser.add_argument('-experiment')
    args = parser.parse_args()

    # move to root dir
    os.chdir(ROOTDIR)

    # global variables
    SEARCH_SPACE = {'lr': [3e-3, 3e-4, 3e-5],
                    'cnn_out_hidden': [256, 512],
                    'cnn_kernel_size': [6, 8],
                    'cnn_n_layers': [2, 3],
                    'z_size': [128],
                    'lstm_hidden_size': [256, 512],
                    'lstm_num_layers': [2, 3],
                    'lstm_dropout': [0.2],
                    'variational_scale': [0.1],
                    'beta': [0.001, 0.0001],
                   }

    hyper_grid = ParameterGrid(SEARCH_SPACE)

    out_path = args.o
    experiment = int(args.experiment)

    experiment_hypers = hyper_grid[experiment]
    experiment_settings = {'out_path': out_path, 'experiment_name': str(experiment)}

    config = configure_config(hypers=experiment_hypers, settings=experiment_settings)

    train_model(config)

    #
    # experiment_batches = [i for i in batched(range(len(hyper_grid)), 5)]
    # for batch in experiment_batches:
    #     write_job_script(experiments=batch,
    #                      experiment_name="vae_pretraining",
    #                      experiment_script="3.0_vae_pretraining.py",
    #                      partition='gpu',
    #                      ntasks='18',
    #                      gpus_per_node=1,
    #                      time="120:00:00"
    #                      )
    #