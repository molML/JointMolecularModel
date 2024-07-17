#!/bin/bash
#SBATCH --job-name=rnn_pretraining_3_4_5
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/rnn_pretraining_3_4_5.out
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

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o $out_path -experiment 3 > "$log_path/${experiment_name}_3.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o $out_path -experiment 4 > "$log_path/${experiment_name}_4.log" &
pid2=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o $out_path -experiment 5 > "$log_path/${experiment_name}_5.log" &
pid3=$!

wait $pid1
wait $pid2
wait $pid3

