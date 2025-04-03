""" Perform inference of all molecules on the AE control model

Derek van Tilborg
Eindhoven University of Technology
Augustus 2024
"""

import os
from os.path import join as ospj
from collections import defaultdict
import pandas as pd
import torch
from tqdm.auto import tqdm
from rdkit import Chem
from cheminformatics.encoding import strip_smiles
from cheminformatics.eval import reconstruction_edit_distance
from cheminformatics.complexity import calculate_bertz_complexity, calculate_molecular_shannon_entropy, \
    calculate_smiles_shannon_entropy
from cheminformatics.molecular_similarity import mean_cosine_cats_to_train, tani_sim_to_train, mcsf_to_train
from jcm.datasets import MoleculeDataset
from jcm.training_logistics import get_all_dataset_names
from constants import ROOTDIR


def load_datasets():

    data_path = ospj('data/split/ChEMBL_33_split.csv')

    # get the train and val SMILES from the pre-processed file
    chembl = pd.read_csv(data_path)
    train_smiles = chembl[chembl['split'] == 'train'].smiles.tolist()
    val_smiles = chembl[chembl['split'] == 'val'].smiles.tolist()
    test_smiles = chembl[chembl['split'] == 'test'].smiles.tolist()

    # Initiate the datasets
    train_dataset = MoleculeDataset(train_smiles, descriptor='smiles', randomize_smiles=False)
    val_dataset = MoleculeDataset(val_smiles, descriptor='smiles', randomize_smiles=False)
    test_dataset = MoleculeDataset(test_smiles, descriptor='smiles', randomize_smiles=False)

    return train_dataset, val_dataset, test_dataset


def do_inference(model, dataset):
    """ Perform inference on a specific dataset, and return a dataframe with inputs, loss, outputs, edit distance """

    predictions = model.predict(dataset, convert_probs_to_smiles=True)
    predicted_smiles = strip_smiles(predictions["token_probs_N_S_C"])

    results = {"predicted_smiles": predicted_smiles,
               "reconstruction_loss": predictions['reconstruction_loss'].cpu(),
               "total_loss": predictions['total_loss'].cpu(),
               'smiles': predictions['smiles'],
               'edit_distance': [reconstruction_edit_distance(i, j) for i, j in zip(predicted_smiles, predictions['smiles'])],
               }

    return pd.DataFrame(results)


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


def calc_distance_metrics(df, outdir):
    # distance to the pretrain set
    distance_metrics = {}
    pretrain_smiles = df[(df.dataset == 'ChEMBL') & (df.split == 'train')].smiles.tolist()
    all_smiles = df.smiles.tolist()

    # Create a mol object for every smiles string for speed
    print('\t\tGenerating molecule library')
    mol_library = {smi: Chem.MolFromSmiles(smi) for smi in tqdm(all_smiles)}

    print('\t\tComputing Tanimoto similarities between ECFPs')
    distance_metrics['Tanimoto_to_pretrain'] = tani_sim_to_train(all_smiles, pretrain_smiles,
                                                                 mol_library=mol_library)
    df = df.assign(**distance_metrics)
    df.to_csv(ospj(outdir, 'all_results_.csv'), index=False)

    distance_metrics['Tanimoto_scaffold_to_train'] = tani_sim_to_train(all_smiles, pretrain_smiles,
                                                                       mol_library=mol_library, scaffold=True)
    df = df.assign(**distance_metrics)
    df.to_csv(ospj(outdir, 'all_results_.csv'), index=False)

    print('\t\tComputing cosine similarities between Cats descriptors')
    distance_metrics['Cats_cos'] = mean_cosine_cats_to_train(all_smiles, pretrain_smiles, mol_library=mol_library)
    df = df.assign(**distance_metrics)
    df.to_csv(ospj(outdir, 'all_results_.csv'), index=False)

    # print('\t\tComputing MCS fraction between molecules')
    # distance_metrics['MCSF'] = mcsf_to_train(all_smiles, pretrain_smiles, mol_library=mol_library)
    # df = df.assign(**distance_metrics)
    # df.to_csv(ospj(outdir, 'all_results_.csv'), index=False)

    return distance_metrics


if __name__ == "__main__":

    BEST_MODEL_PATH = ospj('data', 'best_model', 'pretrained', 'ae', 'model.pt')

    # move to root dir and create a 'best_model' dir to save evaluations
    os.chdir(ROOTDIR)
    outdir = ospj('results', 'ae_pretraining', 'best_model')
    os.makedirs(outdir, exist_ok=True)

    # 1. Get the best model from pretraining
    print(f"Loading best model ...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(BEST_MODEL_PATH, map_location=torch.device(device))
    model.to(device)

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

    df_all = pd.read_csv(ospj(outdir, 'all_results.csv'))
    df_all = df_all.sample(frac=0.1, replace=False, random_state=1)

    # 6. Add distance metrics
    distance_metrics = calc_distance_metrics(df_all, outdir)
    df_all = df_all.assign(**distance_metrics)
    df_all.to_csv(ospj(outdir, 'all_results_.csv'), index=False)
