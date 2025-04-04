""" Perform model inference for the jcm model

Derek van Tilborg
Eindhoven University of Technology
November 2024
"""

import os
from os.path import join as ospj
import argparse
from itertools import batched
import pandas as pd
from jcm.training_logistics import get_all_dataset_names
from constants import ROOTDIR
from jcm.models import JMM
from jcm.datasets import MoleculeDataset
import torch
from sklearn.model_selection import train_test_split
from jcm.utils import logits_to_pred
from cheminformatics.encoding import strip_smiles, probs_to_smiles
from cheminformatics.eval import smiles_validity, reconstruction_edit_distance
from cheminformatics.molecular_similarity import compute_z_distance_to_train


def find_seeds(dataset: str) -> tuple[int]:

    df = pd.read_csv(ospj(BEST_MLPS_ROOT_PATH, dataset, 'results_preds.csv'))

    return tuple(set(df.seed))


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


def perform_inference(model, dataset, train_dataset, seed, library_name):

    predictions = model.predict(dataset)
    keys_to_remove = []
    for k, v in predictions.items():
        if v is None:
            keys_to_remove.append(k)

        if torch.is_tensor(v):
            predictions[k] = v.cpu()

    # actually remove the keys
    for k in keys_to_remove:
        predictions.pop(k)

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
    predictions.update({'seed': seed, 'reconstructed_smiles': reconst_smiles, 'library_name': library_name,
                        'design': designs_clean, 'edit_distance': edit_dist, 'y_hat': y_hat, 'y_unc': y_unc,
                        'y_E': y_E, 'mean_z_dist': mean_z_dist})

    df = pd.DataFrame(predictions)

    print(f'df: {df.shape}')

    return df


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


def write_job_script(dataset_names: list[str], experiment_name: str = "smiles_mlp",
                     experiment_script: str = "4.3_smiles_mlp.py", partition: str = 'gpu', ntasks: str = '18',
                     gpus_per_node: str = 1, time: str = "4:00:00") -> None:
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

    for i, exp in enumerate(dataset_names):
        lines.append('\n')
        lines.append('$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -dataset EX > "$log_path/XE.log" &\n'.replace('EX', str(exp)).replace('XE', f"{experiment_name}_{exp}"))
        lines.append(f'pid{i+1}=$!\n')

    lines.append('\n')
    for i, exp in enumerate(dataset_names):
        lines.append(f'wait $pid{i+1}\n')
    lines.append('\n')

    # Write the modified lines back to the file
    with open(ospj(ROOTDIR, 'experiments', 'jobs', jobname + '.sh'), 'w') as file:
        file.writelines(lines)


if __name__ == '__main__':

    os.chdir(ROOTDIR)

    MODEL = JMM
    EXPERIMENT_NAME = "jmm_library_inference"
    BEST_MLPS_ROOT_PATH = f"/projects/prjs1021/JointChemicalModel/results/smiles_mlp"
    JMM_ROOT_PATH = f"/projects/prjs1021/JointChemicalModel/results/smiles_jmm"

    SPECS_PATH = "data/screening_libraries/specs_cleaned.csv"
    ASINEX_PATH = "data/screening_libraries/asinex_cleaned.csv"
    ENAMINE_HIT_LOCATOR_PATH = "data/screening_libraries/enamine_hit_locator_cleaned.csv"

    # JMM_ROOT_PATH = "results/jmm_CHEMBL233_Ki"

    # Load libraries
    library_specs = MoleculeDataset(pd.read_csv(SPECS_PATH)['smiles_cleaned'].tolist(),
                                    descriptor='smiles', randomize_smiles=False)

    library_asinex = MoleculeDataset(pd.read_csv(ASINEX_PATH)['smiles_cleaned'].tolist(),
                                     descriptor='smiles', randomize_smiles=False)

    library_enamine_hit_locator = MoleculeDataset(pd.read_csv(ENAMINE_HIT_LOCATOR_PATH)['smiles_cleaned'].tolist(),
                                                  descriptor='smiles', randomize_smiles=False)

    libraries = {'asinex': library_asinex,
                 'enamine_hit_locator': library_enamine_hit_locator,
                 'specs': library_specs}

    all_datasets = get_all_dataset_names()

    # experiment_batches = [i for i in batched(range(len(all_datasets)), 1)]
    # for batch in experiment_batches:
    #     dataset_names = [all_datasets[exp_i] for exp_i in batch]
    #
    #     write_job_script(dataset_names=dataset_names,
    #                      experiment_name=EXPERIMENT_NAME,
    #                      experiment_script="6.1_jmm_inference_libraries.py",
    #                      partition='gpu',
    #                      ntasks='18',
    #                      gpus_per_node=1,
    #                      time="24:00:00"
    #                      )

    # parse script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', help='The path of the output directory', default='results')
    parser.add_argument('-dataset')
    args = parser.parse_args()

    dataset_name = args.dataset

    # 2. Find which seeds were used during pretraining. Train a model for every cross-validation split/seed
    seeds = find_seeds(dataset_name)
    print(seeds)

    for library_name, library in libraries.items():

        all_inference_results = []

        for seed in seeds:
            print(f"library: {library_name} - seed: {seed}")

            try:
                # 2.2. get the data belonging to a certain cross-validation split/seed
                train_dataset, val_dataset, test_dataset, ood_dataset = load_data_for_seed(dataset_name, seed)

                # 2.3. load model and setup the device
                model = torch.load(os.path.join(JMM_ROOT_PATH, dataset_name, f"model_{seed}.pt"))
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                model.to(device)
                model.encoder.device = model.decoder.device = model.mlp.device = model.device = device
                model.pretrained_decoder = None

                all_inference_results.append(perform_inference(model, library, train_dataset, seed, library_name))

                # Save to file
                os.makedirs(ospj(JMM_ROOT_PATH, 'screening_libraries'), exist_ok=True)
                pd.concat(all_inference_results).to_csv(ospj(JMM_ROOT_PATH, 'screening_libraries', f'{dataset_name}_{library_name}_inference.csv'), index=False)

            except Exception as error:
                print("An exception occurred:", error)
