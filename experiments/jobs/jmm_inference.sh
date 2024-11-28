#!/bin/bash
#SBATCH --job-name=jmm_inference
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/jmm_inference.out
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00

project_path="$HOME/projects/JointChemicalModel"
experiment_script_path="$project_path/experiments/5.3_inference_jmm.py"

log_path="$project_path/results/logs"

source $HOME/anaconda3/etc/profile.d/conda.sh
export PYTHONPATH="$PYTHONPATH:$project_path"

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path > "$log_path/jmm_inference.log" &
