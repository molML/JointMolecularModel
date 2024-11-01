""" Perform hyperparameter tuning and model training for an SMILES + MLP control model

Derek van Tilborg
Eindhoven University of Technology
July 2024
"""

import os
from os.path import join as ospj
import argparse
from itertools import batched
from tqdm import tqdm
from jcm.config import Config, load_settings, save_settings
from jcm.training_logistics import prep_outdir, get_all_dataset_names, mlp_hyperparam_tuning, nn_cross_validate
from constants import ROOTDIR
from jcm.models import SmilesMLP
from jcm.callbacks import mlp_callback


def write_job_script(dataset_names: list[str], out_paths: list[str] = 'results', experiment_name: str = "smiles_mlp",
                     experiment_script: str = "4.5_smiles_mlp.py", partition: str = 'gpu', ntasks: str = '18',
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


if __name__ == '__main__':

    MODEL = SmilesMLP
    CALLBACK = mlp_callback
    EXPERIMENT_NAME = "smiles_mlp"
    DEFAULT_SETTINGS_PATH = "experiments/hyperparams/smiles_mlp_default.yml"
    HYPERPARAM_GRID = {'mlp_hidden_dim': [1024, 2048],
                       'mlp_n_layers': [2, 3, 4, 5],
                       'lr': [3e-4, 3e-5, 3e-6],
                       'data_augmentation': [False],
                       'cnn_out_hidden': [256, 512],
                       'cnn_kernel_size': [6],
                       'cnn_n_layers': [2, 3],
                       'z_size': [128],
                       }

    # move to root dir
    os.chdir(ROOTDIR)

    all_datasets = get_all_dataset_names()

    # experiment_batches = [i for i in batched(range(len(all_datasets)), 5)]
    # for batch in experiment_batches:
    #     dataset_names = [all_datasets[exp_i] for exp_i in batch]
    #     out_paths = [f"results/{EXPERIMENT_NAME}/{dataset_name}" for dataset_name in dataset_names]
    #
    #     write_job_script(dataset_names=dataset_names,
    #                      out_paths=out_paths,
    #                      experiment_name=EXPERIMENT_NAME,
    #                      experiment_script="4.5_smiles_mlp.py",
    #                      partition='gpu',
    #                      ntasks='18',
    #                      gpus_per_node=1,
    #                      time="80:00:00"
    #                      )

    # parse script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', help='The path of the output directory', default='results')
    parser.add_argument('-dataset')
    args = parser.parse_args()

    out_path = args.o
    dataset_name = args.dataset

    # perform the experiment ###########################################################################################

    best_hypers = mlp_hyperparam_tuning(MODEL, CALLBACK, dataset_name, DEFAULT_SETTINGS_PATH, HYPERPARAM_GRID)

    settings = load_settings(DEFAULT_SETTINGS_PATH)
    config_dict = settings['training_config'] | {'dataset_name': dataset_name, 'out_path': out_path}
    hyperparameters = settings['hyperparameters'] | best_hypers

    config = Config(**config_dict)
    config.set_hyperparameters(**hyperparameters)

    # make output dir
    prep_outdir(config)

    # save best hypers
    save_settings(config, ospj(config.out_path, 'experiment_settings.yml'))

    # perform model training with cross validation and save results
    results = nn_cross_validate(MODEL, CALLBACK, config)
