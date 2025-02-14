""" Perform model inference for the jmm model

Derek van Tilborg
Eindhoven University of Technology
November 2024
"""

import os
from os.path import join as ospj
import pandas as pd
from jmm.training_logistics import get_all_dataset_names
from constants import ROOTDIR
from jmm.models import JMM
from jmm.datasets import MoleculeDataset
import torch
from sklearn.model_selection import train_test_split
from jmm.utils import logits_to_pred
from cheminformatics.encoding import strip_smiles, probs_to_smiles
from cheminformatics.eval import smiles_validity, reconstruction_edit_distance
from cheminformatics.molecular_similarity import compute_z_distance_to_train


def find_seeds(dataset: str) -> tuple[int]:

    df = pd.read_csv(ospj(BEST_MLPS_ROOT_PATH, dataset, 'results_preds.csv'))

    return tuple(set(df.seed))


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


def reconstruct_smiles(logits_N_S_C, true_smiles: list[str]):

    # reconstruction
    designs = probs_to_smiles(logits_N_S_C)

    # Clean designs
    designs_clean = strip_smiles(designs)
    validity, reconstructed_smiles = smiles_validity(designs_clean, return_invalids=True)

    edit_distances = []
    for true_smi, smi in zip(true_smiles, designs_clean):
        edist = reconstruction_edit_distance(true_smi, smi) if smi is not None else None
        edit_distances.append(edist)

    return reconstructed_smiles, designs_clean, edit_distances, validity


def perform_inference(model, train_dataset, test_dataset, ood_dataset, seed):

    def infer(dataset, split: str):

        # perform predictions on all splits, take the average values (sampling from the vae gives different outcomes every
        # time
        predictions = model.predict(dataset)

        for k, v in predictions.items():
            if torch.is_tensor(v):
                predictions[k] = v.cpu()

        # convert y hat logits into binary predictions
        y_hat, y_unc = logits_to_pred(predictions['y_logprobs_N_K_C'], return_binary=True)

        y_E = torch.mean(torch.exp(predictions['y_logprobs_N_K_C']), dim=1)[:, 1]

        # Compute z distances to the train set (not the most efficient but ok)
        mean_z_dist = compute_z_distance_to_train(model, dataset, train_dataset)

        # reconstruct the smiles
        reconst_smiles, designs_clean, edit_dist, validity = reconstruct_smiles(predictions['token_probs_N_S_C'],
                                                                                predictions['smiles'])

        # logits_N_S_C = predictions['token_probs_N_S_C']
        predictions.pop('y_logprobs_N_K_C')
        predictions.pop('token_probs_N_S_C')
        predictions.update({'seed': seed, 'split': split, 'reconstructed_smiles': reconst_smiles,
                            'design': designs_clean, 'edit_distance': edit_dist, 'y_hat': y_hat, 'y_unc': y_unc,
                            'y_E': y_E, 'mean_z_dist': mean_z_dist})

        df = pd.DataFrame(predictions)

        return df

    print("performing inference")
    predictions_train = infer(train_dataset, 'train')
    predictions_test = infer(test_dataset, 'test')
    predictions_ood = infer(ood_dataset, 'ood')

    results_df = pd.concat((predictions_train, predictions_test, predictions_ood))

    return results_df


if __name__ == '__main__':

    os.chdir(ROOTDIR)

    MODEL = JMM
    EXPERIMENT_NAME = "smiles_jmm"
    BEST_MLPS_ROOT_PATH = f"/projects/prjs1021/JointChemicalModel/results/smiles_mlp"
    JMM_ROOT_PATH = f"/projects/prjs1021/JointChemicalModel/results/smiles_jmm"

    all_datasets = get_all_dataset_names()

    for dataset in all_datasets:
        print(dataset)

        if 'ChEMBL_33' not in dataset:

            all_results = []

            # 2. Find which seeds were used during pretraining. Train a model for every cross-validation split/seed
            seeds = find_seeds(dataset)
            print(seeds)
            for seed in seeds:

                try:
                    # 2.2. get the data belonging to a certain cross-validation split/seed
                    train_dataset, val_dataset, test_dataset, ood_dataset = load_data_for_seed(dataset, seed)

                    # 2.3. load model and setup the device
                    model = torch.load(os.path.join(JMM_ROOT_PATH, dataset, f"model_{seed}.pt"))
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    model.to(device)
                    model.encoder.device = model.decoder.device = model.mlp.device = model.device = device
                    if model.pretrained_decoder is not None:
                        model.pretrained_decoder.device = device

                    all_results.append(perform_inference(model, train_dataset, test_dataset, ood_dataset, seed))
                    pd.concat(all_results).to_csv(ospj(JMM_ROOT_PATH, dataset, 'results_preds.csv'), index=False)

                except Exception as error:
                    print("An exception occurred:", error)
