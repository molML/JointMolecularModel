""" Run a model for hyperparameter tuning. This script contains a function to write the specific SLURM scripts I use on
our computer cluster, that in turn run this script with a certain set of hyperparameters

Derek van Tilborg
Eindhoven University of Technology
March 2024
"""

import os
from os.path import join as ospj
from itertools import batched
import pandas as pd
import torch
from sklearn.model_selection import ParameterGrid
from jcm.callbacks import ae_callback
from jcm.config import Config, load_settings, init_experiment, finish_experiment
from jcm.datasets import MoleculeDataset
from jcm.models import AE
from jcm.training import Trainer
from constants import ROOTDIR
import argparse


def load_datasets(config):

    data_path = ospj('data/split/ChEMBL_33_split_prospective.csv')

    # get the train and val SMILES from the pre-processed file
    chembl = pd.read_csv(data_path)
    train_smiles = chembl[chembl['split'] == 'train'].smiles.tolist()
    val_smiles = chembl[chembl['split'] == 'val'].smiles.tolist()

    # Initiate the datasets
    train_dataset = MoleculeDataset(train_smiles, descriptor='smiles', randomize_smiles=config.data_augmentation)
    val_dataset = MoleculeDataset(val_smiles, descriptor='smiles', randomize_smiles=config.data_augmentation)

    return train_dataset, val_dataset


def train_model(config):
    """ Train a model according to the config

    :param config: Config object containing all settings and hypers
    """
    train_dataset, val_dataset = load_datasets(config)

    model = AE(config)

    T = Trainer(config, model, train_dataset, val_dataset)
    if val_dataset is not None:
        T.set_callback('on_batch_end', ae_callback)
    T.run()

    # Save the final model
    torch.save(model, ospj(T.outdir, f"model.pt"))


def write_job_script(out_path: str = 'results', experiment_name: str = "ae_pretraining_extra_data",
                     experiment_script: str = "7.4_ae_pretraining_extra_data.py", partition: str = 'gpu_a100', ntasks: str = '18',
                     gpus_per_node: str = 1, time: str = "72:00:00") -> None:

    jobname = experiment_name

    lines = []
    lines.append('#!/bin/bash\n')
    lines.append(f'#SBATCH --job-name={jobname}\n')
    lines.append(f'#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/{jobname}.out\n')
    lines.append(f'#SBATCH -p {partition}\n')
    lines.append('#SBATCH -N 1\n')
    lines.append(f'#SBATCH --ntasks={ntasks}\n')
    lines.append(f'#SBATCH --gpus-per-node={gpus_per_node}\n')
    lines.append(f'#SBATCH --time={time}\n')
    lines.append('\n')
    lines.append('project_path="$HOME/projects/JointChemicalModel"\n')
    lines.append(f'experiment_script_path="$project_path/experiments/{experiment_script}"\n')
    lines.append('\n')
    lines.append('log_path="$project_path/results/logs"\n')
    lines.append('\n')
    lines.append('source $HOME/anaconda3/etc/profile.d/conda.sh\n')
    lines.append('export PYTHONPATH="$PYTHONPATH:$project_path"\n')

    lines.append('$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o OUT_PATH > "$log_path/XE.log"\n'.replace('XE', f"{experiment_name}").replace('OUT_PATH', out_path))

    lines.append('\n')

    # Move all output files to the project directory
    source = f"$project_path/{out_path}"
    destination = f"/projects/prjs1021/JointChemicalModel/{os.path.dirname(out_path)}/"

    lines.append(f'cp -r {source} {destination}\n')
    lines.append(f"if [ $? -eq 0 ]; then\n    rm -rf {source}\nfi\n\n")

    # Write the modified lines back to the file
    with open(ospj(ROOTDIR, 'experiments', 'jobs', jobname + '.sh'), 'w') as file:
        file.writelines(lines)


if __name__ == '__main__':

    # global variables
    DEFAULT_SETTINGS_PATH = "experiments/hyperparams/ae_pretrain_default.yml"
    EXPERIMENT_NAME = "ae_pretraining_extra_data"
    HYPERPARAMETERS = {'lr': [3e-4],
                    'cnn_out_hidden': [512],
                    'cnn_kernel_size': [6],
                    'cnn_n_layers': [2],
                    'cnn_dropout': [0.1],
                    'z_size': [128],
                    'rnn_type': ['lstm'],
                    'rnn_hidden_size': [512],
                    'rnn_num_layers': [3],
                    'rnn_dropout': [0],
                    'grad_norm_clip': [5],
                    'weight_decay': [0.0001],
                    'data_augmentation': [False],
                    'rnn_teacher_forcing': [False]
                   }

    # write_job_script(out_path=f"results/{EXPERIMENT_NAME}",
    #                  experiment_name=EXPERIMENT_NAME,
    #                  experiment_script="7.4_ae_pretraining_extra_data.py",
    #                  partition='gpu',
    #                  ntasks='18',
    #                  gpus_per_node=1,
    #                  time="120:00:00"
    #                  )

    # # parse script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', help='The path of the output directory', default='results')
    parser.add_argument('-experiment')
    args = parser.parse_args()

    # move to root dir
    os.chdir(ROOTDIR)

    out_path = args.o
    experiment = int(args.experiment)

    experiment_settings = {'out_path': out_path, 'experiment_name': EXPERIMENT_NAME,
                           'data_augmentation': HYPERPARAMETERS['data_augmentation']}

    config = init_experiment(config_path=DEFAULT_SETTINGS_PATH, config_dict=experiment_settings,
                             hyperparameters=HYPERPARAMETERS, name=EXPERIMENT_NAME,
                             group='ae_pretraining')

    print('Experiment config:')
    print(config, '\n')

    train_model(config)

    finish_experiment()
