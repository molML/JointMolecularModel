""" Perform model finetuning for the ae model, just like we did with the JMM model

Derek van Tilborg
Eindhoven University of Technology
November 2024
"""

import os
from os.path import join as ospj
import argparse
from itertools import batched
import pandas as pd
from jcm.config import finish_experiment
from jcm.training import Trainer
from jcm.training_logistics import get_all_dataset_names, prep_outdir
from constants import ROOTDIR
from jcm.models import AE
from jcm.datasets import MoleculeDataset
from jcm.config import init_experiment, load_settings
from jcm.callbacks import ae_callback
import torch
from sklearn.model_selection import train_test_split
from cheminformatics.encoding import strip_smiles, probs_to_smiles
from cheminformatics.eval import smiles_validity, reconstruction_edit_distance


def write_job_script(dataset_names: list[str], out_paths: list[str] = 'results', experiment_name: str = "ae_finetuning",
                     experiment_script: str = "4.7_ae_finetuning.py", partition: str = 'gpu', ntasks: str = '18',
                     gpus_per_node: str = 1, time: str = "120:00:00") -> None:
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

    for i, (exp, out_path) in enumerate(zip(dataset_names, out_paths)):
        lines.append('\n')
        lines.append('$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o OUT_PATH -dataset EX > "$log_path/XE.log" &\n'.replace('EX', str(exp)).replace('XE', f"{experiment_name}_{exp}").replace('OUT_PATH', out_path))
        lines.append(f'pid{i+1}=$!\n')

    lines.append('\n')
    for i, exp in enumerate(dataset_names):
        lines.append(f'wait $pid{i+1}\n')
    lines.append('\n')

    # Move all output files to the project directory
    for i, out_path in enumerate(out_paths):
        source = f"$project_path/{out_path}"
        destination = f"/projects/prjs1021/JointChemicalModel/{os.path.dirname(out_path)}/"

        lines.append(f'cp -r {source} {destination}\n')
        lines.append(f"if [ $? -eq 0 ]; then\n    rm -rf {source}\nfi\n\n")

    # Write the modified lines back to the file
    with open(ospj(ROOTDIR, 'experiments', 'jobs', jobname + '.sh'), 'w') as file:
        file.writelines(lines)


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


def run_models(hypers: dict, out_path: str, experiment_name: str, dataset: str, save_best_model: bool = True):

    best_val_losses = []
    all_results = []

    # 2. Find which seeds were used during pretraining. Train a model for every cross-validation split/seed
    seeds = find_seeds(dataset)
    for seed in seeds:
        # break
        # 2.2. get the data belonging to a certain cross-validation split/seed
        train_dataset, val_dataset, test_dataset, ood_dataset = load_data_for_seed(dataset, seed)

        ae_config = init_experiment(BEST_AE_CONFIG_PATH,
                                    config_dict={'experiment_name': experiment_name, 'out_path': out_path},
                                    hyperparameters=hypers,
                                    group="AE_finetuning",
                                    tags=[str(seed), dataset],
                                    name=experiment_name)

        # 2.3. init model and experiment
        model = torch.load(BEST_AE_MODEL_PATH, map_location=torch.device(ae_config.device))
        model.to(ae_config.device)

        # 2.4. train the model
        T = Trainer(ae_config, model, train_dataset, val_dataset, save_models=False)
        if val_dataset is not None:
            T.set_callback('on_batch_end', ae_callback)
        T.run()

        # 2.5. save model and training history
        if save_best_model:
            torch.save(model, ospj(out_path, f"model_{seed}.pt"))
        if out_path is not None:
            T.get_history(ospj(out_path, f"training_history_{seed}.csv"))

            all_results.append(perform_inference(model, train_dataset, test_dataset, ood_dataset, seed))
            pd.concat(all_results).to_csv(ospj(out_path, 'results_preds.csv'), index=False)

        best_val_losses.append(min(T.history['val_loss']))

    return sum(best_val_losses)/len(best_val_losses)


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

        # reconstruct the smiles
        reconst_smiles, designs_clean, edit_dist, validity = reconstruct_smiles(predictions['token_probs_N_S_C'],
                                                                                predictions['smiles'])

        predictions.pop('token_probs_N_S_C')
        predictions.update({'seed': seed, 'split': split, 'reconstructed_smiles': reconst_smiles,
                            'design': designs_clean, 'edit_distance': edit_dist})

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

    MODEL = AE
    CALLBACK = ae_callback
    EXPERIMENT_NAME = "ae_finetuning"
    BEST_AE_CONFIG_PATH = ospj('data', 'best_model', 'pretrained', 'ae', 'config.yml')
    BEST_AE_MODEL_PATH = ospj('data', 'best_model', 'pretrained', 'ae', 'model.pt')
    BEST_MLPS_ROOT_PATH = f"/projects/prjs1021/JointChemicalModel/results/smiles_mlp"
    BEST_MLPS_ROOT_PATH = "results/smiles_mlp"

    HYPERPARAMS = {'lr': 3e-6,
                   'lr_decoder': 3e-7,
                   'weight_decay': 0}

    all_datasets = get_all_dataset_names()

    # experiment_batches = [tuple(str(j) for j in i) for i in batched(all_datasets, 5)]
    # for batch in experiment_batches:
    #     out_paths = [f"results/{EXPERIMENT_NAME}/{dataset}" for dataset in batch]
    #
    #     write_job_script(dataset_names=batch,
    #                      out_paths=out_paths,
    #                      experiment_name=EXPERIMENT_NAME,
    #                      experiment_script="4.6_jmm_ae_encoder.py",
    #                      partition='gpu',
    #                      ntasks='18',
    #                      gpus_per_node=1,
    #                      time="120:00:00"
    #                      )

    # parse script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', help='The path of the output directory', default='results')
    parser.add_argument('-dataset')
    args = parser.parse_args()

    out_path = args.o
    dataset = args.dataset

    os.makedirs(out_path, exist_ok=True)

    # Train the JMM model with the best hyperparameters, but now save the models
    run_models(HYPERPARAMS, out_path=out_path, experiment_name=f"{EXPERIMENT_NAME}_{dataset}",
               dataset=dataset, save_best_model=True)

    finish_experiment()
