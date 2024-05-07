
import os
from os.path import join as ospj
import random
import pandas as pd
import torch
from sklearn.model_selection import ParameterGrid
from jcm.datasets import MoleculeDataset
from jcm.trainer import train_lstm_vae
from jcm.config import Config, load_settings


DEFAULT_SETTINGS_PATH = "experiments/hyperparams/vae_pretrain_default.yml"
TUNEABLE_HYPERS = {'lr': [3e-3, 3e-4],
                   'kernel_size': [6, 8],
                   'hidden_dim_lstm': [256, 512],
                   'hidden_dim_cnn': [256, 512],
                   'n_layers_cnn': [2, 3],
                   'learnable_cell_state': [False, True],
                   'variational_scale': [0.1, 0.01],
                   'beta': [0.0001, 0.001],
                   }


PATH_TRAIN_SMILES = ospj('data', 'ChEMBL', 'chembl_train_smiles.csv')
PATH_VAL_SMILES = ospj('data', 'ChEMBL', 'chembl_val_smiles.csv')


def get_hyper_grid(default_settings_path, tuneable_hypers, log_file: str = None):

    settings_grid = []
    for experiment, hypers in enumerate(ParameterGrid(tuneable_hypers)):
        experiment_settings = load_settings(default_settings_path)
        experiment_settings['training_config']['experiment_name'] = f"pretrain_{experiment}"
        experiment_settings['training_config']['out_path'] = ospj(experiment_settings['training_config']['out_path'], 'pretrain_vae_hyperopt')
        experiment_settings['hyperparameters'] = experiment_settings['hyperparameters'] | hypers
        settings_grid.append(experiment_settings)

    # shuffle it so it doesn't take forever before we reach certain hypers
    random.seed(0)
    random.shuffle(settings_grid)

    df = pd.DataFrame([d['training_config'] | d['hyperparameters'] for d in settings_grid])
    if log_file is not None:
        df.to_csv(log_file, index=False)

    return settings_grid


def run_model(settings, overwrite: bool = False):

    hypers = settings['hyperparameters']
    config = settings['training_config']
    PATH_OUT = ospj(config['out_path'], config['experiment_name'])

    try:
        os.makedirs(PATH_OUT, exist_ok=overwrite)

        train_smiles = pd.read_csv(PATH_TRAIN_SMILES).smiles.tolist()
        train_dataset = MoleculeDataset(train_smiles, descriptor=config['descriptor'], randomize_smiles=True)

        val_smiles = pd.read_csv(PATH_VAL_SMILES).smiles.tolist()
        val_dataset = MoleculeDataset(val_smiles, descriptor=config['descriptor'], randomize_smiles=True)

        config_ = Config(**config)
        config_.set_hyperparameters(**hypers)

        model, trainer = train_lstm_vae(config_, train_dataset, val_dataset)

        del model, trainer

    except:
        pass


def get_best_hypers():
    df = merge_hypertuning_results()

    results_dfs_last_iter = df[df['iter_num'] == 49500]
    best_hyper_row = results_dfs_last_iter.sort_values(by=['val_loss']).head(1)

    best_hypers = {'lr': best_hyper_row['lr'].item(),
                   'kernel_size': best_hyper_row['kernel_size'].item(),
                   'hidden_dim_lstm': best_hyper_row['hidden_dim_lstm'].item(),
                   'hidden_dim_cnn': best_hyper_row['hidden_dim_cnn'].item(),
                   'n_layers_cnn': best_hyper_row['n_layers_cnn'].item(),
                   'learnable_cell_state': best_hyper_row['learnable_cell_state'].item(),
                   'variational_scale': best_hyper_row['variational_scale'].item(),
                   'beta': best_hyper_row['beta'].item()}

    return best_hypers


def merge_hypertuning_results():
    results = [i for i in os.listdir('results/pretrain_vae_hyperopt') if i.startswith('pretrain')]
    results = [ospj('results', 'pretrain_vae_hyperopt', i, 'training_history.csv') for i in results if
               os.path.exists(ospj('results', 'pretrain_vae_hyperopt', i, 'training_history.csv'))]
    settings_file = pd.read_csv('results/pretrain_vae_hyperopt/pre_training_hyperopt.csv')

    results_dfs = []
    for result_path in results:
        try:
            df = pd.read_csv(result_path)
            settings_row = settings_file[settings_file.experiment_name == result_path.split('/')[2]]
            settings_df = pd.concat([settings_row] * len(df), ignore_index=True)
            results_dfs.append(pd.concat([df, settings_df], axis='columns'))
        except:
            pass
    results_dfs = pd.concat(results_dfs, axis='rows')
    results_dfs.to_csv('results/pretrain_vae_hyperopt/hyperparameter_tuning_results.csv')

    return results_dfs


def pretrain_model(hypers: dict, experiment_name: str, max_iters: int = 200000, save_every: int = 1000):

    # configure the training settings
    pretrain_settings = load_settings(DEFAULT_SETTINGS_PATH)
    pretrain_settings['training_config']['experiment_name'] = experiment_name
    pretrain_settings['hyperparameters'] = pretrain_settings['hyperparameters'] | hypers
    pretrain_config = pretrain_settings['training_config']
    pretrain_hypers = pretrain_settings['hyperparameters']
    pretrain_config['max_iters'] = max_iters
    pretrain_config['save_every'] = save_every
    pretrain_config['val_molecules_to_sample'] = 10000

    # make the output dir
    os.makedirs(ospj(pretrain_config['out_path'], pretrain_config['experiment_name']), exist_ok=True)

    # setup the datasets
    train_smiles = pd.read_csv(PATH_TRAIN_SMILES).smiles.tolist()
    train_dataset = MoleculeDataset(train_smiles, descriptor=pretrain_config['descriptor'], randomize_smiles=True)
    val_smiles = pd.read_csv(PATH_VAL_SMILES).smiles.tolist()
    val_dataset = MoleculeDataset(val_smiles, descriptor=pretrain_config['descriptor'], randomize_smiles=True)

    # Train the model
    config_ = Config(**pretrain_config)
    config_.set_hyperparameters(**pretrain_hypers)
    model, trainer = train_lstm_vae(config_, train_dataset, val_dataset)

    # Save the full model (so not just the state dict)
    torch.save(model, ospj(config_.out_path, config_.experiment_name, f"pre_trained_model.pt"))


if __name__ == '__main__':

    settings_grid = get_hyper_grid(DEFAULT_SETTINGS_PATH, TUNEABLE_HYPERS,
                                   'results/pretrain_vae_hyperopt/pre_training_hyperopt.csv')

    for settings in settings_grid:
        print(settings)
        run_model(settings, overwrite=True)

    print('Hyperparameter tuning is done')

    # Merge all tuning results files into one big file and extract the best hyperparameters from it
    best_hypers = get_best_hypers()

    # train the pre-trained model with the best hypers. We let it train a little bit longer
    pretrain_model(best_hypers, experiment_name="pretrained_vae2", max_iters=100000, save_every=1000)

    print('Model pretraining is done')
