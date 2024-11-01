""" Perform model training for the JVAE model

Derek van Tilborg
Eindhoven University of Technology
September 2024
"""

import os
import random
from os.path import join as ospj
import argparse
from itertools import batched
from collections import defaultdict
import numpy as np
import pandas as pd
from jcm.config import finish_experiment
from jcm.training import Trainer
from jcm.training_logistics import get_all_dataset_names, prep_outdir
from constants import ROOTDIR
from jcm.models import JMM
from jcm.datasets import MoleculeDataset
from jcm.config import init_experiment, load_settings
from jcm.callbacks import jmm_callback
import torch
from sklearn.model_selection import train_test_split, ParameterGrid
from jcm.utils import logits_to_pred
from cheminformatics.encoding import strip_smiles, probs_to_smiles
from cheminformatics.eval import smiles_validity, reconstruction_edit_distance, plot_molecular_reconstruction


def write_job_script(dataset_names: list[str], out_paths: list[str] = 'results', experiment_name: str = "jvae",
                     experiment_script: str = "4.5_jvae.py", partition: str = 'gpu', ntasks: str = '18',
                     gpus_per_node: str = 1, time: str = "120:00:00") -> None:
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


def find_seeds(dataset: str) -> tuple[int]:

    df = pd.read_csv(ospj(BEST_MLPS_ROOT_PATH, dataset, 'results_preds.csv'))

    return tuple(set(df.seed))


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
    mlp_config_path = ospj(BEST_MLPS_ROOT_PATH, dataset, 'experiment_settings.yml')

    # update config for the JVAE
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    jvae_config = jvae_merge_pretrained_configs(default_config_path, best_vae_config_path, mlp_config_path,
                                                hyperparameters=hyperparameters | {'device': device},
                                                training_config=training_config)

    jvae_config = init_experiment(jvae_config, launch_wandb=False)
    return jvae_config


def init_jvae(jvae_config, best_vae_weights_path: str, dataset: str, seed: int):

    mlp_model_path = ospj(BEST_MLPS_ROOT_PATH, dataset, f"model_{seed}.pt")

    # init JVAE model
    model = JMM(jvae_config)

    # load pretrained VAE weights
    model.load_vae_weights(best_vae_weights_path)

    # load pretrained MLP weights
    model.load_mlp_weights(get_mlp_state_dict(mlp_model_path, jvae_config))
    model.to(jvae_config.device)

    return model


def run_models(hypers: dict, out_path: str, experiment_name: str, dataset: str, save_best_model: bool = True):

    best_val_losses = []
    all_results = []

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
                                      group="jvae",
                                      tags=[str(seed), dataset],
                                      name=experiment_name)

        # 2.4. train the model
        T = Trainer(jvae_config, model, train_dataset, val_dataset, save_models=False)
        if val_dataset is not None:
            T.set_callback('on_batch_end', jmm_callback)
        T.run()

        # 2.5. save model and training history
        if save_best_model:
            model.save_weights(ospj(out_path, f"model_{seed}.pt"))
        if out_path is not None:
            T.get_history(ospj(out_path, f"training_history_{seed}.csv"))
            model.load_pretrained(BEST_VAE_PATH)  # load the pretrained VAE to unbias the finetuned VAE

            all_results.append(perform_inference(model, train_dataset, test_dataset, ood_dataset, seed))
            pd.concat(all_results).to_csv(ospj(out_path, 'results_preds.csv'), index=False)

        best_val_losses.append(min(T.history['val_loss']))

    return sum(best_val_losses)/len(best_val_losses)


def reconstruct_smiles(logits_N_S_C, true_smiles: list[str]):

    # reconstruction
    designs = probs_to_smiles(logits_N_S_C)

    # Clean designs
    designs_clean = strip_smiles(designs)
    validity, reconstructed_smiles = smiles_validity(designs_clean, return_invalids=True)

    edit_distances = []
    for true_smi, smi in zip(true_smiles, reconstructed_smiles):
        edist = reconstruction_edit_distance(true_smi, smi) if smi is not None else None
        edit_distances.append(edist)

    return reconstructed_smiles, edit_distances, validity


def perform_inference(model, train_dataset, test_dataset, ood_dataset, seed):

    # perform predictions on all splits
    predictions_train = model.predict(train_dataset)
    logits_N_K_C_token_train = predictions_train['token_probs_N_S_C']
    y_logits_N_K_C_train = predictions_train['y_logprobs_N_K_C']
    y_train, smiles_train = predictions_train['y'], predictions_train['smiles']

    predictions_test = model.predict(test_dataset)
    logits_N_K_C_token_test = predictions_test['token_probs_N_S_C']
    y_logits_N_K_C_test = predictions_test['y_logprobs_N_K_C']
    y_test, smiles_test = predictions_test['y'], predictions_test['smiles']

    predictions_ood = model.predict(ood_dataset)
    logits_N_K_C_token_ood = predictions_ood['token_probs_N_S_C']
    y_logits_N_K_C_ood = predictions_ood['y_logprobs_N_K_C']
    y_ood, smiles_ood = predictions_ood['y'], predictions_ood['smiles']

    smiles_train, molecule_reconstruction_losses_train = model.ood_score(train_dataset)
    smiles_test, molecule_reconstruction_losses_test = model.ood_score(test_dataset)
    smiles_ood, molecule_reconstruction_losses_ood = model.ood_score(ood_dataset)

    # convert y hat logits into binary predictions
    y_hat_train, y_unc_train = logits_to_pred(y_logits_N_K_C_train, return_binary=True)
    y_hat_test, y_unc_test = logits_to_pred(y_logits_N_K_C_test, return_binary=True)
    y_hat_ood, y_unc_ood = logits_to_pred(y_logits_N_K_C_ood, return_binary=True)

    # reconstruct the smiles
    reconst_smiles_train, edit_dist_train, validity_train = reconstruct_smiles(logits_N_K_C_token_train, smiles_train)
    reconst_smiles_test, edit_dist_test, validity_test = reconstruct_smiles(logits_N_K_C_token_test, smiles_test)
    reconst_smiles_ood, edit_dist_ood, validity_ood = reconstruct_smiles(logits_N_K_C_token_ood, smiles_ood)

    # Put the predictions in a dataframe
    train_results_df = pd.DataFrame({'seed': seed, 'split': 'train', 'smiles': smiles_train,
                                     'reconstructed_smiles': reconst_smiles_train,
                                     'edit_distance': edit_dist_train,
                                     'reconstruction_loss': molecule_reconstruction_losses_train.cpu(),
                                     'y': y_train.cpu(), 'y_hat': y_hat_train.cpu(),
                                     'y_unc': y_unc_train.cpu()})

    # Put the predictions in a dataframe
    test_results_df = pd.DataFrame({'seed': seed, 'split': 'test', 'smiles': smiles_test,
                                     'reconstructed_smiles': reconst_smiles_test,
                                     'edit_distance': edit_dist_test,
                                     'reconstruction_loss': molecule_reconstruction_losses_test.cpu(),
                                     'y': y_test.cpu(), 'y_hat': y_hat_test.cpu(),
                                     'y_unc': y_unc_test.cpu()})

    # Put the predictions in a dataframe
    ood_results_df = pd.DataFrame({'seed': seed, 'split': 'ood', 'smiles': smiles_ood,
                                     'reconstructed_smiles': reconst_smiles_ood,
                                     'edit_distance': edit_dist_ood,
                                     'reconstruction_loss': molecule_reconstruction_losses_ood.cpu(),
                                     'y': y_ood.cpu(), 'y_hat': y_hat_ood.cpu(),
                                     'y_unc': y_unc_ood.cpu()})

    results_df = pd.concat((train_results_df, test_results_df, ood_results_df))

    return results_df


if __name__ == '__main__':

    os.chdir(ROOTDIR)

    MODEL = JMM
    CALLBACK = jmm_callback
    EXPERIMENT_NAME = "jvae"
    DEFAULT_SETTINGS_PATH = "experiments/hyperparams/jvae_default.yml"
    BEST_VAE_PATH = ospj('data', 'best_model', 'pretrained', 'vae', 'weights.pt')
    BEST_VAE_CONFIG_PATH = ospj('data', 'best_model', 'pretrained', 'vae', 'config.yml')
    BEST_MLPS_ROOT_PATH = f"/projects/prjs1021/JointChemicalModel/results/smiles_var_mlp"

    HYPERPARAMS = {'lr': 3e-5, 'mlp_loss_scalar': 0.1, 'freeze_encoder': False}

    # SEARCH_SPACE = {'lr': [3e-5],               # lr seems to be the most important for accuracy and edit distance
    #                 'mlp_loss_scalar': [0.1],   # didn't seem to matter that much, this puts it in the same order of magnitude as the reconstruction loss
    #                 'freeze_encoder': [False],  # didn't seem to impact performance
    #                 }
    # hyper_grid = ParameterGrid(SEARCH_SPACE)

    all_datasets = get_all_dataset_names()

    # experiment_batches = [tuple(str(j) for j in i) for i in batched(all_datasets, 5)]
    # for batch in experiment_batches:
    #     out_paths = [f"results/{EXPERIMENT_NAME}/{dataset}" for dataset in batch]
    #
    #     write_job_script(dataset_names=batch,
    #                      out_paths=out_paths,
    #                      experiment_name=EXPERIMENT_NAME,
    #                      experiment_script="4.5_jvae.py",
    #                      partition='gpu',
    #                      ntasks='18',
    #                      gpus_per_node=1,
    #                      time="120:00:00"
    #                      )

    # # parse script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', help='The path of the output directory', default='results')
    parser.add_argument('-dataset')
    args = parser.parse_args()

    out_path = args.o
    dataset = args.dataset

    # Train models in 10-fold cross validation over the whole hyperparameter space.
    # hyper_performance = defaultdict(list)
    # for exp_i, hypers in enumerate(hyper_grid):
    #
    #     # create an experiment-specific experiment_name
    #     _experiment_name = f"{EXPERIMENT_NAME}_{dataset}_{exp_i}"
    #
    #     mean_val_loss = run_models(hypers, out_path=None, experiment_name=_experiment_name, dataset=dataset,
    #                                save_best_model=False)
    #
    #     hyper_performance['mean_val_loss'].append(mean_val_loss)
    #     hyper_performance['hypers'].append(hypers)
    #
    # # Get the best performing hyperparameters
    # best_hypers = hyper_performance['hypers'][np.argmin(hyper_performance['mean_val_loss'])]
    # print(f"\n\nBest hyperparams (val loss of {min(hyper_performance['mean_val_loss']):.4f}) are:\n{best_hypers}\n\n")

    os.makedirs(out_path, exist_ok=True)

    # Train the JVAE model with the best hyperparameters, but now save the models
    run_models(HYPERPARAMS, out_path=out_path, experiment_name=f"{EXPERIMENT_NAME}_{dataset}",
               dataset=dataset, save_best_model=True)

    perform_inference(out_path=out_path)

    finish_experiment()
