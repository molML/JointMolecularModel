#!/bin/bash
#SBATCH --job-name=jmm_library_inference_CHEMBL1862_Ki_CHEMBL214_Ki_CHEMBL2047_EC50
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/jmm_library_inference_CHEMBL1862_Ki_CHEMBL214_Ki_CHEMBL2047_EC50.out
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00

project_path="$HOME/projects/JointChemicalModel"
experiment_script_path="$project_path/experiments/6.1_jmm_inference_libraries.py"

log_path="$project_path/results/logs"

source $HOME/anaconda3/etc/profile.d/conda.sh
export PYTHONPATH="$PYTHONPATH:$project_path"

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -dataset CHEMBL1862_Ki > "$log_path/jmm_library_inference_CHEMBL1862_Ki.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -dataset CHEMBL214_Ki > "$log_path/jmm_library_inference_CHEMBL214_Ki.log" &
pid2=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -dataset CHEMBL2047_EC50 > "$log_path/jmm_library_inference_CHEMBL2047_EC50.log" &
pid3=$!

wait $pid1
wait $pid2
wait $pid3

