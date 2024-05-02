from jcm.utils import logits_to_smiles
from eval.design_eval import strip_smiles, smiles_validity, reconstruction_edit_distance
from dataprep.descriptors import mols_to_ecfp
from dataprep.utils import smiles_to_mols
from rdkit.Chem import DataStructs
import pandas as pd

from os.path import join as ospj

import torch

from jcm.datasets import MoleculeDataset
from jcm.trainer import train_lstm_vae
from jcm.config import Config, load_settings
from jcm.utils import load_model
from jcm.model import LstmVAE
from sklearn.model_selection import ParameterGrid


DEFAULT_SETTINGS_PATH = "experiments/hyperparams/vae_pretrain_default.yml"
PATH_TRAIN_SMILES = ospj('data', 'ChEMBL', 'chembl_train_smiles.csv')
PATH_VAL_SMILES = ospj('data', 'ChEMBL', 'chembl_val_smiles.csv')
PATH_TEST_SMILES = ospj('data', 'ChEMBL', 'chembl_test_smiles.csv')


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

    model = torch.load("results/pretrained_vae/pre_trained_model.pt")

    df_train, zs_train = evaluate_pretrain_reconstructions(model, train_dataset)
    df_val, zs_val = evaluate_pretrain_reconstructions(model, val_dataset)
    df_test, zs_tets = evaluate_pretrain_reconstructions(model, test_dataset)


    df_val.to_csv('validation_reconstruction.csv')
    df_test.to_csv('test_reconstruction.csv')

    count = [smi for smi in train_smiles if ':' in smi]
    len(count)
    print('done')

