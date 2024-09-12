""" Perform model training for the JVAE model

Derek van Tilborg
Eindhoven University of Technology
Augustus 2024
"""

import os
from os.path import join as ospj
import argparse
from itertools import batched

import pandas as pd
from tqdm import tqdm
from jcm.config import Config, load_settings, save_settings, finish_experiment
from jcm.training import Trainer
from jcm.training_logistics import prep_outdir, get_all_dataset_names, mlp_hyperparam_tuning, nn_cross_validate
from constants import ROOTDIR
from jcm.models import JointChemicalModel as JVAE, VAE, SmilesMLP
from jcm.datasets import load_datasets, MoleculeDataset
from jcm.config import load_and_setup_config_from_file, init_experiment, load_settings
from jcm.callbacks import jvae_callback
import torch
from sklearn.model_selection import train_test_split


def write_job_script(dataset_names: list[str], out_paths: list[str] = 'results', experiment_name: str = "cats_mlp",
                     experiment_script: str = "4.2_cats_mlp.py", partition: str = 'gpu', ntasks: str = '18',
                     gpus_per_node: str = 1, time: str = "4:00:00") -> None:
    """
    :param experiments: list of experiment numbers, e.g. [0, 1, 2]
    """

    jobname = experiment_name + '_' + '_'.join([str(i) for i in dataset_names])

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

    for i, (exp, out_path) in enumerate(zip(dataset_names, out_paths)):
        lines.append('\n')
        lines.append('$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o OUT_PATH -dataset EX > "$log_path/XE.log" &\n'.replace('EX', str(exp)).replace('XE', f"{experiment_name}_{exp}").replace('OUT_PATH', out_path))
        lines.append(f'pid{i+1}=$!\n')

    lines.append('\n')
    for i, exp in enumerate(dataset_names):
        lines.append(f'wait $pid{i+1}\n')
    lines.append('\n')

    # Move all output files to the project directory
    for i, out_path in enumerate(out_paths):
        source = f"$project_path/{out_path}"
        destination = f"/projects/prjs1021/JointChemicalModel/{os.path.dirname(out_path)}/"

        lines.append(f'cp -r {source} {destination}\n')
        lines.append(f"if [ $? -eq 0 ]; then\n    rm -rf {source}\nfi\n\n")

    # Write the modified lines back to the file
    with open(ospj(ROOTDIR, 'experiments', 'jobs', jobname + '.sh'), 'w') as file:
        file.writelines(lines)


def jvae_merge_pretrained_configs(default_config_path: str, vae_config_path: str, mlp_config_path: str,
                                  hyperparameters: dict = None, training_config: dict = None):

    default_config = load_settings(default_config_path)
    vae_config = load_settings(vae_config_path)
    mlp_config = load_settings(mlp_config_path)

    default_config['hyperparameters'].update(mlp_config['hyperparameters'])
    default_config['hyperparameters'].update(vae_config['hyperparameters'])
    if hyperparameters is not None:
        default_config['hyperparameters'].update(hyperparameters)

    default_config['training_config'].update(mlp_config['training_config'])
    default_config['training_config'].update(vae_config['training_config'])
    default_config['training_config'].update(load_settings(default_config_path)['training_config'])
    if training_config is not None:
        default_config['training_config'].update(training_config)

    return default_config


def find_seeds(dataset: str) -> list[int]:

    df = pd.read_csv(ospj("data", "best_model", "smiles_mlp", dataset, 'results_preds.csv'))

    return set(df.seed)


def get_mlp_state_dict(mlp_model_path: str, config):
    pretrained_mlp = torch.load(mlp_model_path, map_location=torch.device(config.device))

    return pretrained_mlp.mlp.state_dict()


def load_data_for_seed(dataset_name: str, seed: int):
    """ load the data splits associated with a specific seed """

    val_size = 0.1

    # get the train and val SMILES from the pre-processed file
    data_path = ospj(f'data/split/{dataset_name}_split.csv')
    data = pd.read_csv(data_path)

    train_data = data[data['split'] == 'train']
    test_data = data[data['split'] == 'test']
    ood_data = data[data['split'] == 'ood']

    train_data, val_data = train_test_split(train_data, test_size=val_size, random_state=seed)

    # Initiate the datasets
    val_dataset = MoleculeDataset(val_data.smiles.tolist(), val_data.y.tolist(),
                                  descriptor='smiles', randomize_smiles=False)

    train_dataset = MoleculeDataset(train_data.smiles.tolist(), train_data.y.tolist(),
                                    descriptor='smiles', randomize_smiles=False)

    test_dataset = MoleculeDataset(test_data.smiles.tolist(), test_data.y.tolist(),
                                   descriptor='smiles', randomize_smiles=False)

    ood_dataset = MoleculeDataset(ood_data.smiles.tolist(), ood_data.y.tolist(),
                                  descriptor='smiles', randomize_smiles=False)

    return train_dataset, val_dataset, test_dataset, ood_dataset


def setup_config(default_config_path: str, best_vae_config_path: str, hyperparameters: dict, training_config: dict):

    # get config for the pretrained MLP
    pretrained_mlp_root_path = ospj("data", "best_model", "smiles_mlp")
    mlp_config_path = ospj(pretrained_mlp_root_path, dataset, 'experiment_settings.yml')

    # update config for the JVAE
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    jvae_config = jvae_merge_pretrained_configs(default_config_path, best_vae_config_path, mlp_config_path,
                                                hyperparameters=hyperparameters | {'device': device},
                                                training_config=training_config)

    jvae_config = init_experiment(jvae_config, launch_wandb=False)
    return jvae_config


def init_jvae(jvae_config, best_vae_weights_path: str, dataset: str, seed: int):

    pretrained_mlp_root_path = ospj("data", "best_model", "smiles_mlp")
    mlp_model_path = ospj(pretrained_mlp_root_path, dataset, f"model_{seed}.pt")

    # init JVAE model
    model = JVAE(jvae_config)

    # load pretrained VAE weights
    model.load_vae_weights(best_vae_weights_path)

    # load pretrained MLP weights
    model.load_mlp_weights(get_mlp_state_dict(mlp_model_path, jvae_config))
    model.to(jvae_config.device)

    return model


if __name__ == '__main__':

    os.chdir(ROOTDIR)

    MODEL = JVAE
    CALLBACK = jvae_callback
    EXPERIMENT_NAME = "jvae"
    DEFAULT_SETTINGS_PATH = "experiments/hyperparams/jvae_default.yml"
    BEST_VAE_PATH = ospj('data', 'best_model', 'pretrained', 'vae', 'weights.pt')
    BEST_VAE_CONFIG_PATH = ospj('data', 'best_model', 'pretrained', 'vae', 'config.yml')

    SEARCH_SPACE = {'lr': [3e-4, 3e-5, 3e-6],
                    'mlp_loss_scalar': [0.01, 0.1, 1],
                    'freeze_encoder': [True, False]}

    # TODO write the training scripts

    dataset = "CHEMBL204_Ki"
    hypers = {'lr': 3e-6, 'mlp_loss_scalar': 0.01, 'freeze_encoder': True}
    out_path = 'results/jvae'
    experiment_name = 'test'

    # 1. Setup config
    jvae_config = setup_config(default_config_path=DEFAULT_SETTINGS_PATH, best_vae_config_path=BEST_VAE_CONFIG_PATH,
                               hyperparameters=hypers,
                               training_config={'out_path': out_path, 'experiment_name': experiment_name})

    # 2. Find which seeds were used during pretraining. Train a model for every cross-validation split/seed
    seeds = find_seeds(dataset)
    for seed in seeds:
        # 2.2. get the data belonging to a certain cross-validation split/seed
        train_dataset, val_dataset, test_dataset, ood_dataset = load_data_for_seed(dataset, seed)

        # 2.3. init model and experiment
        model = init_jvae(jvae_config, best_vae_weights_path=BEST_VAE_PATH, dataset=dataset, seed=seed)
        jvae_config = init_experiment(jvae_config,
                                      group="finetuning_tryout",
                                      tags=[str(seed), dataset],
                                      name=str(experiment_name))

        # 2.4. train the model
        T = Trainer(jvae_config, model, train_dataset, val_dataset)
        if val_dataset is not None:
            T.set_callback('on_batch_end', jvae_callback)
        T.run()

        # 2.5. save model
        model.save_weights(ospj(out_path, f"model_{seed}.pt"))

    finish_experiment()

