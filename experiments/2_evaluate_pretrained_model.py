

import os
from os.path import join as ospj
import pandas as pd
from umap import UMAP
from rdkit.Chem import DataStructs
import torch
from jcm.utils import logits_to_smiles
from eval.design_eval import strip_smiles, smiles_validity, reconstruction_edit_distance
from cheminformatics.descriptors import mols_to_ecfp
from cheminformatics.utils import smiles_to_mols
from jcm.datasets import MoleculeDataset


DEFAULT_SETTINGS_PATH = "experiments/hyperparams/vae_pretrain_default.yml"
PATH_TRAIN_SMILES = ospj('data', 'ChEMBL', 'chembl_train_smiles.csv')
PATH_VAL_SMILES = ospj('data', 'ChEMBL', 'chembl_val_smiles.csv')
PATH_TEST_SMILES = ospj('data', 'ChEMBL', 'chembl_test_smiles.csv')
PATH_MOLECULEACE = ospj('data', 'moleculeace')


def evaluate_pretrain_reconstructions(model, dataset) -> (pd.DataFrame, torch.Tensor):
    """ Evaluates the reconstruction abilities of our pre-trained model

    :param model: LstmVAE model
    :param dataset: a dataset
    :return: dataframe with: 'smiles', 'reconstructed_smiles', 'valid_smiles', 'likelihood',
             'edit_distance', 'tanimoto' and a tensor of latent representations
    """

    x_hats, zs, sample_likelihoods = model.predict(dataset)

    # Convert logits to SMILES strings
    raw_designs = logits_to_smiles(x_hats)
    designs = strip_smiles(raw_designs)

    # Get the valid SMILES
    validity, valid_smiles = smiles_validity(designs, return_invalids=True)

    # Compute the edit distance
    edist = [reconstruction_edit_distance(p_smi, t_smi) for p_smi, t_smi in zip(designs, dataset.smiles)]

    # Compute Tanimoto similarity
    tani = []
    for target_smi, reconstr_smi in zip(dataset.smiles, valid_smiles):
        if reconstr_smi is None:
            tani.append(None)
        else:
            fps = mols_to_ecfp(smiles_to_mols([target_smi, reconstr_smi]))
            tani.append(DataStructs.TanimotoSimilarity(fps[0], fps[1]))

    # put it all together in a dataframe
    df = pd.DataFrame({'smiles': dataset.smiles,
                       'reconstructed_smiles': designs,
                       'valid_smiles': valid_smiles,
                       'likelihood': sample_likelihoods.tolist(),
                       'edit_distance': edist,
                       'tanimoto': tani})

    return df, zs


if __name__ == '__main__':

    # setup the datasets
    train_smiles = pd.read_csv(PATH_TRAIN_SMILES).smiles.tolist()
    train_dataset = MoleculeDataset(train_smiles, descriptor='smiles')

    val_smiles = pd.read_csv(PATH_VAL_SMILES).smiles.tolist()
    val_dataset = MoleculeDataset(val_smiles, descriptor='smiles')

    test_smiles = pd.read_csv(PATH_TEST_SMILES).smiles.tolist()
    test_dataset = MoleculeDataset(test_smiles, descriptor='smiles')

    moleculeace_smiles = []
    for filename in os.listdir(PATH_MOLECULEACE):
        moleculeace_smiles.extend(pd.read_csv(f"{PATH_MOLECULEACE}/{filename}").smiles.tolist())
    moleculeace_dataset = MoleculeDataset(list(set(moleculeace_smiles)), descriptor='smiles')

    # Load the model
    model = torch.load("results/pretrained_vae/pre_trained_model.pt")

    # evaluate the model
    df_train, zs_train = evaluate_pretrain_reconstructions(model, train_dataset)
    df_val, zs_val = evaluate_pretrain_reconstructions(model, val_dataset)
    df_test, zs_tests = evaluate_pretrain_reconstructions(model, test_dataset)
    df_moleculeace, zs_moleculeace = evaluate_pretrain_reconstructions(model, moleculeace_dataset)

    # Add a column specifying the split
    df_train['split'] = 'train'
    df_val['split'] = 'val'
    df_test['split'] = 'test'
    df_moleculeace['split'] = 'moleculeace'

    # merge all data into one thing
    df_all = pd.concat([df_train, df_test, df_val, df_moleculeace], axis='rows')
    zs_all = torch.cat((zs_train, zs_val, zs_tests, zs_moleculeace), 0)

    # U-MAP
    umap = UMAP(n_components=2, n_jobs=1, n_neighbors=50, min_dist=0.1)
    embedding = umap.fit_transform(zs_all.cpu().numpy())
    df_all['umap1'] = embedding[:, 0]
    df_all['umap2'] = embedding[:, 1]

    # Save df
    df_all.to_csv('results/pretrained_vae/pretrained_reconstruction.csv', index=False)

    print('done')
