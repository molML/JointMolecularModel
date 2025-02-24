#!/bin/bash
#SBATCH --job-name=library_inference_asinex
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/library_inference_asinex.out
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=1
#SBATCH --time=20:00:00

project_path="$HOME/projects/JointChemicalModel"
experiment_script_path="$project_path/experiments/6.2_jmm_inference_asinex.py"

log_path="$project_path/results/logs"

source $HOME/anaconda3/etc/profile.d/conda.sh
export PYTHONPATH="$PYTHONPATH:$project_path"

echo 'running script'
$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path > "$log_path/library_inference_asinex.log"
