
import pandas as pd
import os
from os.path import join as ospj
from jcm.datasets import MoleculeDataset
from jcm.trainer import train_lstm_vae
from jcm.config import Config, load_settings
from sklearn.model_selection import ParameterGrid


DEFAULT_SETTINGS_PATH = "experiments/hyperparams/vae_pretrain_default.yml"
TUNEABLE_HYPERS = {'lr': [3e-3, 3e-4, 3e-5],
                  'kernel_size': [6, 8, 10],
                  'hidden_dim_lstm': [256, 512],
                  'hidden_dim_cnn': [256],
                  'n_layers_cnn': [2, 3],
                  'learnable_cell_state': [True, False],
                  'variational_scale': [0.1, 0.01],
                  'beta': [0.001, 0.0001]
                  }


def get_hyper_grid(default_settings_path, tuneable_hypers, log_file: str = None):

    settings_grid = []
    for experiment, hypers in enumerate(ParameterGrid(tuneable_hypers)):
        experiment_settings = load_settings(default_settings_path)
        experiment_settings['training_config']['experiment_name'] = f"pretrain_{experiment}"
        experiment_settings['hyperparameters'] = experiment_settings['hyperparameters'] | hypers
        settings_grid.append(experiment_settings)

    df = pd.DataFrame([d['training_config'] | d['hyperparameters'] for d in settings_grid])
    if log_file is not None:
        df.to_csv(log_file, index=False)

    return settings_grid


def run_model(settings, overwrite: bool = False):

    HYPERS = settings['hyperparameters']
    CONFIG = settings['training_config']

    PATH_TRAIN_SMILES = ospj('data', 'ChEMBL', 'chembl_train_smiles.csv')
    PATH_VAL_SMILES = ospj('data', 'ChEMBL', 'chembl_val_smiles.csv')
    PATH_OUT = ospj(CONFIG['out_path'], CONFIG['experiment_name'])

    try:
        os.makedirs(PATH_OUT, exist_ok=overwrite)

        train_smiles = pd.read_csv(PATH_TRAIN_SMILES).smiles.tolist()
        train_dataset = MoleculeDataset(train_smiles, descriptor=CONFIG['descriptor'])

        val_smiles = pd.read_csv(PATH_VAL_SMILES).smiles.tolist()
        val_dataset = MoleculeDataset(val_smiles, descriptor=CONFIG['descriptor'])

        config = Config(**CONFIG)
        config.set_hyperparameters(**HYPERS)

        model, trainer = train_lstm_vae(config, train_dataset, val_dataset)

        del model, trainer

    except:
        pass


if __name__ == '__main__':

    settings_grid = get_hyper_grid(DEFAULT_SETTINGS_PATH, TUNEABLE_HYPERS, 'results/pre_training_hyperopt.csv')

    for settings in settings_grid:
        print(settings)
        run_model(settings, overwrite=False)

    print('done')

    # Merge all tuning results files into one big file

    results = [i for i in os.listdir('results') if i.startswith('pretrain')]
    results = [ospj('results', i, 'training_history.csv') for i in results if os.path.exists(ospj('results', i, 'training_history.csv'))]
    settings_file = pd.read_csv('results/pre_training_hyperopt.csv')

    results_dfs = []
    for result_path in results:
        df = pd.read_csv(result_path)
        settings_row = settings_file[settings_file.experiment_name == result_path.split('/')[1]]
        settings_df = pd.concat([settings_row] * len(df), ignore_index=True)
        results_dfs.append(pd.concat([df, settings_df], axis='columns'))
    results_dfs = pd.concat(results_dfs, axis='rows')

    results_dfs.to_csv('results/hyperparameter_tuning_results.csv')