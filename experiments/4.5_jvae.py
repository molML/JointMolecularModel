""" Perform model training for the JVAE model

Derek van Tilborg
Eindhoven University of Technology
Augustus 2024
"""

import os
from os.path import join as ospj
import argparse
from itertools import batched
from tqdm import tqdm
from jcm.config import Config, load_settings, save_settings
from jcm.training_logistics import prep_outdir, get_all_dataset_names, mlp_hyperparam_tuning, nn_cross_validate
from constants import ROOTDIR
from jcm.models import JointChemicalModel as JVAE
from jcm.callbacks import jvae_callback


def write_job_script(dataset_names: list[str], out_paths: list[str] = 'results', experiment_name: str = "cats_mlp",
                     experiment_script: str = "4.2_cats_mlp.py", partition: str = 'gpu', ntasks: str = '18',
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

    MODEL = JVAE
    CALLBACK = jvae_callback
    EXPERIMENT_NAME = "jvae"
    DEFAULT_SETTINGS_PATH = "experiments/hyperparams/jvae_default.yml"

    # Hyperparams
    # Lr, Loss scalar, freezing VAE, freezing MLP, pre-trained MLP, pre-trained VAE

