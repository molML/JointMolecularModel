#!/bin/bash
#SBATCH --job-name=jmm_library_inference_PPARG_CHEMBL244_Ki_CHEMBL204_Ki_CHEMBL218_EC50_CHEMBL233_Ki
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/jmm_library_inference_PPARG_CHEMBL244_Ki_CHEMBL204_Ki_CHEMBL218_EC50_CHEMBL233_Ki.out
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

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -dataset PPARG > "$log_path/jmm_library_inference_PPARG.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -dataset CHEMBL244_Ki > "$log_path/jmm_library_inference_CHEMBL244_Ki.log" &
pid2=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -dataset CHEMBL204_Ki > "$log_path/jmm_library_inference_CHEMBL204_Ki.log" &
pid3=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -dataset CHEMBL218_EC50 > "$log_path/jmm_library_inference_CHEMBL218_EC50.log" &
pid4=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -dataset CHEMBL233_Ki > "$log_path/jmm_library_inference_CHEMBL233_Ki.log" &
pid5=$!

wait $pid1
wait $pid2
wait $pid3
wait $pid4
wait $pid5

