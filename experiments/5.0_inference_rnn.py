""" Perform inference of all molecules on the RNN control model

Derek van Tilborg
Eindhoven University of Technology
Augustus 2024
"""

import os
from os.path import join as ospj
from collections import defaultdict
import pandas as pd
from tqdm.auto import tqdm
from rdkit import Chem
import torch
from constants import ROOTDIR
from cheminformatics.encoding import strip_smiles
from cheminformatics.eval import smiles_validity, reconstruction_edit_distance, uniqueness, novelty
from jcm.config import load_and_setup_config_from_file
from jcm.training_logistics import get_all_dataset_names
from jcm.datasets import MoleculeDataset
from jcm.models import DeNovoRNN
from cheminformatics.complexity import calculate_bertz_complexity, calculate_molecular_shannon_entropy, \
    calculate_smiles_shannon_entropy, count_unique_motifs


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


def load_model(config_path: str, state_dict_path: str):
    """ Load an RNN model from disk given its config and state dict"""

    # load the model settings
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = load_and_setup_config_from_file(config_path, hyperparameters={'device': device})  # set the device to be sure

    # Load the model
    model = DeNovoRNN(config)
    model.load_state_dict(torch.load(state_dict_path, map_location=torch.device(device)))
    model.to(device)

    return model


def do_inference(model, dataset):
    """ Perform inference on a specific dataset, and return a dataframe with inputs, loss, outputs, edit distance """

    predictions = model.predict(dataset, convert_probs_to_smiles=True)
    predicted_smiles = strip_smiles(predictions['token_probs_N_S_C'])

    results = {"predicted_smiles": predicted_smiles,
               "reconstruction_loss": predictions["reconstruction_loss"].cpu(),
               'smiles': predictions["smiles"],
               'edit_distance': [reconstruction_edit_distance(i, j) for i, j in zip(predicted_smiles, predictions["smiles"])],
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


def add_complexity_metrics(df):
    complexity = defaultdict(list)
    for smi in tqdm(df.smiles.tolist()):
        mol = Chem.MolFromSmiles(smi)

        complexity['Bertz'].append(calculate_bertz_complexity(mol))
        complexity['molecule_entropy'].append(calculate_molecular_shannon_entropy(mol))
        complexity['smiles_entropy'].append(calculate_smiles_shannon_entropy(smi))

    # Adding the dictionary as new columns
    df = df.assign(**complexity)

    return df


if __name__ == "__main__":

    BEST_MODEL_WEIGHTS = ospj('data', 'best_model', 'pretrained', 'rnn', 'weights.pt')
    BEST_MODEL_CONFIG = ospj('data', 'best_model', 'pretrained', 'rnn', 'config.yml')

    # move to root dir and create a 'best_model' dir to save evaluations
    os.chdir(ROOTDIR)
    outdir = ospj('results', 'rnn_pretraining', 'best_model')
    os.makedirs(outdir, exist_ok=True)

    # 1. Load the best model from pretraining
    print(f"Loading best model ...")

    model = load_model(config_path=BEST_MODEL_CONFIG, state_dict_path=BEST_MODEL_WEIGHTS)

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

    # 5. Add complexity metrics and save results.
    df_all = pd.concat([df_chembl, df_datasets])
    df_all = add_complexity_metrics(df_all)
    df_all.to_csv(ospj(outdir, 'all_results.csv'), index=False)
