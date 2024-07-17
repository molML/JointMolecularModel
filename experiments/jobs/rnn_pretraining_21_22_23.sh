#!/bin/bash
#SBATCH --job-name=rnn_pretraining_21_22_23
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/rnn_pretraining_21_22_23.out
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=1
#SBATCH --time=120:00:00

experiment_name="rnn_pretraining"

project_path="$HOME/projects/JointChemicalModel"
experiment_script_path="$project_path/experiments/3.1_rnn_pretraining.py"

out_path="$project_path/results/$experiment_name"
log_path="$project_path/results/logs"

source $HOME/anaconda3/etc/profile.d/conda.sh
export PYTHONPATH="$PYTHONPATH:$project_path"

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o $out_path -experiment 21 > "$log_path/${experiment_name}_21.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o $out_path -experiment 22 > "$log_path/${experiment_name}_22.log" &
pid2=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o $out_path -experiment 23 > "$log_path/${experiment_name}_23.log" &
pid3=$!

wait $pid1
wait $pid2
wait $pid3

