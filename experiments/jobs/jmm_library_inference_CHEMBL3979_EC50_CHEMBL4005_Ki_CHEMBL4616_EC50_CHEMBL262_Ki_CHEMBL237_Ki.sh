#!/bin/bash
#SBATCH --job-name=jmm_library_inference_CHEMBL3979_EC50_CHEMBL4005_Ki_CHEMBL4616_EC50_CHEMBL262_Ki_CHEMBL237_Ki
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/jmm_library_inference_CHEMBL3979_EC50_CHEMBL4005_Ki_CHEMBL4616_EC50_CHEMBL262_Ki_CHEMBL237_Ki.out
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

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -dataset CHEMBL3979_EC50 > "$log_path/jmm_library_inference_CHEMBL3979_EC50.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -dataset CHEMBL4005_Ki > "$log_path/jmm_library_inference_CHEMBL4005_Ki.log" &
pid2=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -dataset CHEMBL4616_EC50 > "$log_path/jmm_library_inference_CHEMBL4616_EC50.log" &
pid3=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -dataset CHEMBL262_Ki > "$log_path/jmm_library_inference_CHEMBL262_Ki.log" &
pid4=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -dataset CHEMBL237_Ki > "$log_path/jmm_library_inference_CHEMBL237_Ki.log" &
pid5=$!

wait $pid1
wait $pid2
wait $pid3
wait $pid4
wait $pid5

