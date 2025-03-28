#!/bin/bash
#SBATCH --job-name=jmm_library_inference_prospective_CHEMBL308_Ki
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/jmm_library_inference_prospective_CHEMBL308_Ki.out
#SBATCH -p gpu_a100
#SBATCH -N 1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00

project_path="$HOME/projects/JointChemicalModel"
experiment_script_path="$project_path/experiments/7.6_inference_jmm_prospective.py"

log_path="$project_path/results/logs"

source $HOME/anaconda3/etc/profile.d/conda.sh
export PYTHONPATH="$PYTHONPATH:$project_path"

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -dataset CHEMBL308_Ki > "$log_path/jmm_library_inference_prospective_CHEMBL308_Ki.log" &
pid1=$!

wait $pid1

