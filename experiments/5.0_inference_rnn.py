""" Perform inference of all molecules on the RNN control model

Derek van Tilborg
Eindhoven University of Technology
Augustus 2024
"""

import os
from os.path import join as ospj
import pandas as pd
from constants import ROOTDIR
from jcm.config import Config, load_settings, load_and_setup_config_from_file
from cheminformatics.encoding import strip_smiles, probs_to_smiles
from cheminformatics.eval import smiles_validity, reconstruction_edit_distance, uniqueness, novelty
from jcm.datasets import MoleculeDataset
from jcm.models import DeNovoRNN
import torch

RNN_RESULTS = ospj('results', 'rnn_pretraining')


def find_best_experiment() -> str:
    # find the best pretrained model based on validation loss
    best_rows_per_exp = []
    experiment_dirs = [i for i in os.listdir(RNN_RESULTS) if not i.startswith('.') and i != 'best_model']
    for exp_name in experiment_dirs:
        # load training history file
        df = pd.read_csv(ospj(RNN_RESULTS, exp_name, 'training_history.csv'))
        df['experiment'] = exp_name

        # Select the row with the minimum value in column 'A'
        min_value_row = df.loc[df['val_loss'].idxmin()]
        best_rows_per_exp.append(dict(min_value_row))

    pretraining_results = pd.DataFrame(best_rows_per_exp).set_index('experiment')

    # Get the experiment with the lowest val loss
    best_experiment = pretraining_results['val_loss'].idxmin()

    return best_experiment


def load_datasets():

    data_path = ospj('data/split/ChEMBL_33_split.csv')

    # get the train and val SMILES from the pre-processed file
    chembl = pd.read_csv(data_path)
    train_smiles = chembl[chembl['split'] == 'train'].smiles.tolist()
    val_smiles = chembl[chembl['split'] == 'val'].smiles.tolist()
    test_smiles = chembl[chembl['split'] == 'val'].smiles.tolist()

    # Initiate the datasets
    train_dataset = MoleculeDataset(train_smiles, descriptor='smiles', randomize_smiles=False)
    val_dataset = MoleculeDataset(val_smiles, descriptor='smiles', randomize_smiles=False)
    test_dataset = MoleculeDataset(test_smiles, descriptor='smiles', randomize_smiles=False)

    return train_dataset, val_dataset, test_dataset


def eval_generative_performance(model, n=100000):
    # Evaluate the generative performance of the model
    designs = model.generate(n=n)

    designs_clean = strip_smiles(designs)
    validity, valid_smiles = smiles_validity(designs_clean, return_invalids=True)

    # compute uniqueness and novelty
    train_dataset, __, _ = load_datasets()

    metrics = {'validity': validity,
               'novelty': novelty(valid_smiles, train_dataset.smiles),
               'uniqueness': uniqueness(valid_smiles)}

    return metrics


if __name__ == "__main__":

    # move to root dir and create a 'best_model' dir to save evaluations
    os.chdir(ROOTDIR)
    outdir = ospj(RNN_RESULTS, 'best_model')
    os.makedirs(outdir, exist_ok=True)

    # 1. Load the model
    best_experiment = find_best_experiment()

    # load the model settings
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = load_and_setup_config_from_file(ospj(RNN_RESULTS, best_experiment, 'experiment_settings.yml'),
                                             hyperparameters={'device': device})  # set the device to be sure
    print(f"Loading config: \n{config}")

    # load the best checkpoint (the last checkpoint is also the best one)
    best_checkpoint = sorted([i for i in os.listdir(ospj(RNN_RESULTS, best_experiment)) if i.startswith('checkp')])[-1]
    best_checkpoint_path = ospj(RNN_RESULTS, best_experiment, best_checkpoint)

    # Load the model
    model = DeNovoRNN(config)
    model.load_state_dict(torch.load(best_checkpoint_path, map_location=torch.device(device)))

    # Compute validity, novelty, uniqueness
    print('Computing generative performance metrics')
    generative_metrics = eval_generative_performance(model, n=100)

    # save general performance metrics
    pd.DataFrame([generative_metrics]).to_csv(ospj(outdir, 'general_metrics.csv'), index=False)


    # 3. Get all data


    train_dataset, val_dataset, test_dataset = load_datasets()

    all_probs, all_sample_losses, all_lossses, all_smiles = model.predict(val_dataset)


    # 4. Perform inference
    # 5. Save results.




