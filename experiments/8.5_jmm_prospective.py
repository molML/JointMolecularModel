""" Perform model training for the jcm model using the AE encoder

Derek van Tilborg
Eindhoven University of Technology
March 2025
"""

import os
from os.path import join as ospj
import argparse
from itertools import batched
from tqdm import tqdm
import pandas as pd
from warnings import warn
from jcm.config import finish_experiment
from jcm.training import Trainer
from constants import ROOTDIR
from jcm.models import JMM
from jcm.datasets import MoleculeDataset
from jcm.config import init_experiment, load_settings
from jcm.callbacks import jmm_callback
import torch
from sklearn.model_selection import train_test_split
from jcm.utils import logits_to_pred
from cheminformatics.encoding import strip_smiles, probs_to_smiles
from cheminformatics.eval import smiles_validity, reconstruction_edit_distance
from sklearn.metrics import balanced_accuracy_score


def write_job_script(dataset_names: list[str], out_paths: list[str] = 'results', experiment_name: str = "jcm",
                     experiment_script: str = "8.5_jmm_prospective.py", partition: str = 'gpu', ntasks: str = '18',
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


def setup_jmm_config(default_jmm_config_path: str, pretrained_ae_config_path: str, pretrained_mlp_config_path: str,
                     pretrained_ae_path: str = None, pretrained_encoder_mlp_path: str = None,
                     hyperparameters: dict = None, training_config: dict = None):

    # setup the paths in the jcm config.
    variational = True if 'vae' in pretrained_ae_config_path or 'var' in pretrained_mlp_config_path else False
    if hyperparameters is None:
        hyperparameters = {}
    hyperparameters.update({'pretrained_ae_path': pretrained_ae_path,
                            'pretrained_encoder_mlp_path': pretrained_encoder_mlp_path,
                            'variational': variational})

    jmm_config = load_settings(default_jmm_config_path)
    jmm_config['hyperparameters'].update(hyperparameters)

    # merge ae and mlp configs
    pretrained_ae_config = load_settings(pretrained_ae_config_path)
    pretrained_mlp_config = load_settings(pretrained_mlp_config_path)
    jmm_config['hyperparameters'].update(pretrained_mlp_config['hyperparameters'])
    jmm_config['hyperparameters'].update(pretrained_ae_config['hyperparameters'])
    # jmm_config['training_config'].update(pretrained_mlp_config['training_config'])
    # jmm_config['training_config'].update(pretrained_ae_config['training_config'])

    # overwrite the hyperparams and training config with the supplied arguments
    if training_config is not None:
        jmm_config['training_config'].update(training_config)
    if hyperparameters is not None:
        jmm_config['hyperparameters'].update(hyperparameters)

    # automatically update this. Usually this happens somewhere else.
    jmm_config['hyperparameters']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    jmm_config = init_experiment(jmm_config, launch_wandb=False)
    print(jmm_config)
    return jmm_config


def find_seeds(dataset: str) -> tuple[int]:

    df = pd.read_csv(ospj(BEST_MLPS_ROOT_PATH, dataset, 'results_preds.csv'))

    return tuple(set(df.seed))


def load_data_for_seed(dataset_name: str, seed: int):
    """ load the data splits associated with a specific seed """

    val_size = 0.1

    # get the train and val SMILES from the pre-processed file
    data_path = ospj(f'data/clean/{dataset_name}.csv')

    # get the train and val SMILES from the pre-processed file
    train_data = pd.read_csv(data_path)

    train_data, val_data = train_test_split(train_data, test_size=val_size, random_state=seed)

    # Initiate the datasets
    val_dataset = MoleculeDataset(val_data.smiles.tolist(), val_data.y.tolist(),
                                  descriptor='smiles', randomize_smiles=False)

    train_dataset = MoleculeDataset(train_data.smiles.tolist(), train_data.y.tolist(),
                                    descriptor='smiles', randomize_smiles=False)

    return train_dataset, val_dataset


def run_models(hypers: dict, out_path: str, experiment_name: str, dataset: str, save_best_model: bool = True,
               libraries: dict = None):

    best_val_losses = []
    all_results = []
    all_metrics = []

    # 2. Find which seeds were used during pretraining. Train a model for every cross-validation split/seed
    seeds = find_seeds(dataset)
    print(seeds)
    for seed in seeds:
        print(seed)

        # 2.2. get the data belonging to a certain cross-validation split/seed
        train_dataset, val_dataset = load_data_for_seed(dataset, seed)

        # setup config
        pretrained_mlp_config_path = ospj(BEST_MLPS_ROOT_PATH, dataset, "experiment_settings.yml")
        pretrained_mlp_model_path = ospj(BEST_MLPS_ROOT_PATH, dataset, f"model_{seed}.pt")

        jmm_config = setup_jmm_config(default_jmm_config_path=DEFAULT_JMM_CONFIG_PATH,
                                      pretrained_ae_config_path=BEST_AE_CONFIG_PATH,
                                      pretrained_ae_path=BEST_AE_MODEL_PATH,
                                      pretrained_mlp_config_path=pretrained_mlp_config_path,
                                      pretrained_encoder_mlp_path=pretrained_mlp_model_path,
                                      hyperparameters=hypers,
                                      training_config={'experiment_name': experiment_name, 'out_path': out_path})

        # 2.3. init model and experiment
        model = JMM(jmm_config)
        model.to(jmm_config.device)

        jmm_config = init_experiment(jmm_config,
                                     group="JMM_prospective",
                                     tags=[str(seed), dataset],
                                     name=experiment_name)

        # 2.4. train the model
        T = Trainer(jmm_config, model, train_dataset, val_dataset, save_models=True)
        if val_dataset is not None:
            T.set_callback('on_batch_end', jmm_callback)
        T.run()

        # 2.5. save model and training history
        if save_best_model:
            torch.save(model, ospj(out_path, f"model_{seed}.pt"))

        T.get_history(ospj(out_path, f"training_history_{seed}.csv"))
        best_val_losses.append(min(T.history['val_loss']))

        print(f"performing inference on train ({seed})")
        train_inference_df = perform_inference(model, train_dataset, 'train', seed)

        print(f"performing inference on val ({seed})")
        val_inference_df = perform_inference(model, val_dataset, 'val', seed)

        # Put the performance metrics in a dataframe
        all_metrics.append({'seed': seed,
                            'train_balanced_acc': balanced_accuracy_score(train_inference_df['y'], train_inference_df['y_hat']),
                            'train_mean_uncertainty': train_inference_df['y_unc'].mean(),
                            'val_balanced_acc': balanced_accuracy_score(val_inference_df['y'], val_inference_df['y_hat']),
                            'val_mean_uncertainty': val_inference_df['y_unc'].mean()
                            })

        all_results.append(train_inference_df)
        all_results.append(val_inference_df)

        if libraries is not None:
            for library_name, library in libraries.items():
                print(f"performing inference on {library_name} ({seed})")

                all_results.append(perform_inference(model, library, library_name, seed))

        pd.concat(all_results).to_csv(ospj(out_path, 'results_preds.csv'), index=False)
        pd.DataFrame(all_metrics).to_csv(ospj(out_path, 'results_metrics.csv'), index=False)

        finish_experiment()


def mean_tensors_in_dict_list(dict_list):

    # Initialize the result with the first dictionary
    result = dict_list[0].copy()

    for key in result:
        first_value = result[key]
        if torch.is_tensor(first_value) and key != 'y':
            # Collect tensor values for this key from all dictionaries
            tensors = [d[key] for d in dict_list]

            # Stack the tensors and compute the mean
            mean_tensor = torch.mean(torch.stack(tensors), dim=0)
            result[key] = mean_tensor
        else:
            # Non-tensor values are taken from the first dictionary
            result[key] = first_value

        if torch.is_tensor(first_value):
            result[key] = result[key].cpu()

    return result


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


def perform_inference(model, dataset, split, seed, n_samples: int = 10):

    if not model.config.variational:
        n_samples = 1

    # perform predictions
    predictions = mean_tensors_in_dict_list([model.predict(dataset) for i in tqdm(range(n_samples))])

    # convert y hat logits into binary predictions
    y_hat, y_unc = logits_to_pred(predictions['y_logprobs_N_K_C'], return_binary=True)
    y_E = torch.mean(torch.exp(predictions['y_logprobs_N_K_C']), dim=1)[:, 1]

    # reconstruct the smiles
    reconst_smiles, designs_clean, edit_dist, validity = reconstruct_smiles(predictions['token_probs_N_S_C'],
                                                                            predictions['smiles'])

    predictions.pop('y_logprobs_N_K_C')
    predictions.pop('token_probs_N_S_C')
    predictions.update({'seed': seed, 'split': split, 'reconstructed_smiles': reconst_smiles, 'design': designs_clean,
                        'edit_distance': edit_dist, 'y_hat': y_hat, 'y_unc': y_unc, 'y_E': y_E})

    df = pd.DataFrame(predictions)

    return df


if __name__ == '__main__':

    os.chdir(ROOTDIR)

    MODEL = JMM
    CALLBACK = jmm_callback
    EXPERIMENT_NAME = "smiles_jmm_prospective"
    DEFAULT_JMM_CONFIG_PATH = "experiments/hyperparams/jmm_default.yml"
    BEST_AE_CONFIG_PATH = ospj('data', 'best_model', 'pretrained', 'ae_prospective', 'config.yml')
    BEST_AE_MODEL_PATH = ospj('data', 'best_model', 'pretrained', 'ae_prospective', 'model.pt')
    BEST_MLPS_ROOT_PATH = f"/projects/prjs1021/JointChemicalModel/results/smiles_mlp_prospective"

    HYPERPARAMS = {'lr': 3e-6,
                   'lr_decoder': 3e-7,
                   'mlp_loss_scalar': 0.1,
                   'weight_decay': 0,
                   'use_ae_encoder': False}

    all_datasets = ['CHEMBL4718_Ki', 'CHEMBL308_Ki', 'CHEMBL2147_Ki']

    SPECS_PATH = "data/screening_libraries/specs_2025/specs_clean_Apr2025.csv"

    # Load libraries
    library_specs = MoleculeDataset(pd.read_csv(SPECS_PATH)['smiles_cleaned'].tolist(),
                                    descriptor='smiles', randomize_smiles=False)
    libraries = {'specs_Apr2025': library_specs}

    # experiment_batches = [tuple(str(j) for j in i) for i in batched(all_datasets, 1)]
    # for batch in experiment_batches:
    #     out_paths = [f"results/{EXPERIMENT_NAME}/{dataset}" for dataset in batch]
    #
    #     write_job_script(dataset_names=batch,
    #                      out_paths=out_paths,
    #                      experiment_name=EXPERIMENT_NAME,
    #                      experiment_script="8.5_jmm_prospective.py",
    #                      partition='gpu_a100',
    #                      ntasks='18',
    #                      gpus_per_node=1,
    #                      time="48:00:00"
    #                      )

    # parse script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', help='The path of the output directory', default='results')
    parser.add_argument('-dataset')
    args = parser.parse_args()

    out_path = args.o
    dataset = args.dataset

    os.makedirs(out_path, exist_ok=True)

    hypers = HYPERPARAMS

    # Train the JMM model with the best hyperparameters, but now save the models
    run_models(HYPERPARAMS, out_path=out_path, experiment_name=f"{EXPERIMENT_NAME}_{dataset}",
               dataset=dataset, save_best_model=True, libraries=libraries)

    finish_experiment()
