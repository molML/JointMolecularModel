#!/bin/bash
#SBATCH --job-name=vae_pretraining_363_364_365
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/vae_pretraining_363_364_365.out
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=1
#SBATCH --time=120:00:00

experiment_name="vae_pretraining"

project_path="$HOME/projects/JointChemicalModel"
experiment_script_path="$project_path/experiments/3.0_vae_pretraining.py"

out_path="$project_path/results/$experiment_name"
log_path="$project_path/results/logs"

source $HOME/anaconda3/etc/profile.d/conda.sh
export PYTHONPATH="$PYTHONPATH:$project_path"

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o $out_path -experiment 363 > "$log_path/${experiment_name}_363.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o $out_path -experiment 364 > "$log_path/${experiment_name}_364.log" &
pid2=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o $out_path -experiment 365 > "$log_path/${experiment_name}_365.log" &
pid3=$!

wait $pid1
wait $pid2
wait $pid3

