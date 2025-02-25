#!/bin/bash
#SBATCH --job-name=jmm_library_inference_CHEMBL231_Ki_CHEMBL264_Ki_ESR1_ant_CHEMBL1871_Ki_TP53
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/jmm_library_inference_CHEMBL231_Ki_CHEMBL264_Ki_ESR1_ant_CHEMBL1871_Ki_TP53.out
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

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -dataset CHEMBL231_Ki > "$log_path/jmm_library_inference_CHEMBL231_Ki.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -dataset CHEMBL264_Ki > "$log_path/jmm_library_inference_CHEMBL264_Ki.log" &
pid2=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -dataset ESR1_ant > "$log_path/jmm_library_inference_ESR1_ant.log" &
pid3=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -dataset CHEMBL1871_Ki > "$log_path/jmm_library_inference_CHEMBL1871_Ki.log" &
pid4=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -dataset TP53 > "$log_path/jmm_library_inference_TP53.log" &
pid5=$!

wait $pid1
wait $pid2
wait $pid3
wait $pid4
wait $pid5

