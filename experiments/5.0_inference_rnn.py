""" Perform inference of all molecules on the RNN control model

Derek van Tilborg
Eindhoven University of Technology
Augustus 2024
"""

import os
from os.path import join as ospj
import pandas as pd
from constants import ROOTDIR
from cheminformatics.encoding import strip_smiles
from cheminformatics.eval import smiles_validity, reconstruction_edit_distance, uniqueness, novelty
from jcm.config import load_and_setup_config_from_file
from jcm.training_logistics import get_all_dataset_names
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


def load_best_model():
    # 1. Load the model
    best_experiment = find_best_experiment()

    # load the model settings
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = load_and_setup_config_from_file(ospj(RNN_RESULTS, best_experiment, 'experiment_settings.yml'),
                                             hyperparameters={'device': device})  # set the device to be sure

    # load the best checkpoint (the last checkpoint is also the best one)
    best_weights = sorted([i for i in os.listdir(ospj(RNN_RESULTS, best_experiment)) if i.startswith('checkp')])[-1]
    best_checkpoint_path = ospj(RNN_RESULTS, best_experiment, best_weights)

    # Load the model
    model = DeNovoRNN(config)
    model.load_state_dict(torch.load(best_checkpoint_path, map_location=torch.device(device)))
    model.to(device)

    return model


def do_inference(model, dataset):
    """ Perform inference on a specific dataset, and return a dataframe with inputs, loss, outputs, edit distance """

    predicted_smiles, all_sample_losses, _, true_smiles = model.predict(dataset, convert_probs_to_smiles=True)
    predicted_smiles = strip_smiles(predicted_smiles)

    results = {"predicted_smiles": predicted_smiles,
               "reconstruction_loss": all_sample_losses.cpu(),
               'smiles': true_smiles,
               'edit_distance': [reconstruction_edit_distance(i, j) for i, j in zip(predicted_smiles, true_smiles)],
               }

    return pd.DataFrame(results)


def inference_on_dataset(model):
    """ perform inference on all 33 datasets """

    all_dataset_names = get_all_dataset_names()
    all_results = []
    for dataset_name in all_dataset_names:
        # get the train and val SMILES from the pre-processed file
        data_path = ospj(f'data/split/{dataset_name}_split.csv')
        data = pd.read_csv(data_path)
        smiles = data.smiles.tolist()

        # turn into dataset
        dataset = MoleculeDataset(smiles, descriptor='smiles', randomize_smiles=False)

        # perform predictions
        results = do_inference(model, dataset)

        # add to original data and append to all_results
        results = pd.merge(data, results, on='smiles', how='left', validate='one_to_one')
        results['dataset'] = dataset_name
        all_results.append(results)

    return pd.concat(all_results)


def inference_on_chembl(model):
    train_dataset, val_dataset, test_dataset = load_datasets()

    # test set
    df_test = do_inference(model, test_dataset)
    df_test['dataset'] = 'ChEMBL'
    df_test['split'] = 'test'

    # train set
    df_train = do_inference(model, train_dataset)
    df_train['dataset'] = 'ChEMBL'
    df_train['split'] = 'train'

    # val set
    df_val = do_inference(model, val_dataset)
    df_val['dataset'] = 'ChEMBL'
    df_val['split'] = 'val'

    return pd.concat([df_test, df_train, df_val])


if __name__ == "__main__":

    # move to root dir and create a 'best_model' dir to save evaluations
    os.chdir(ROOTDIR)
    outdir = ospj(RNN_RESULTS, 'best_model')
    os.makedirs(outdir, exist_ok=True)

    # 1. Get the best model from pretraining
    print(f"Loading best model ...")
    model = load_best_model()

    # # 2. Compute and save validity, novelty, uniqueness metrics
    # print('Computing generative performance metrics ...')
    # generative_metrics = eval_generative_performance(model, n=100000)
    # pd.DataFrame([generative_metrics]).to_csv(ospj(outdir, 'general_metrics.csv'), index=False)

    # 3. Inference on ChEMBLv33 (all in-distribution data)
    print('Performing inference on ChEMBLv33 (might take a while) ...')
    df_chembl = inference_on_chembl(model)

    # 4. Inference on all predictive datasets (these are all out-of-distribution for this model by design)
    print('Performing inference on all datasets (might take a while) ...')
    df_datasets = inference_on_dataset(model)

    # 5. Save results.
    df_all = pd.concat([df_chembl, df_datasets])
    df_all.to_csv(ospj(outdir, 'all_results.csv'), index=False)
