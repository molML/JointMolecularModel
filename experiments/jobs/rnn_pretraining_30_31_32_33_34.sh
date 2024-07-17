#!/bin/bash
#SBATCH --job-name=rnn_pretraining_30_31_32_33_34
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/rnn_pretraining_30_31_32_33_34.out
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

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o $out_path -experiment 30 > "$log_path/${experiment_name}_30.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o $out_path -experiment 31 > "$log_path/${experiment_name}_31.log" &
pid2=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o $out_path -experiment 32 > "$log_path/${experiment_name}_32.log" &
pid3=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o $out_path -experiment 33 > "$log_path/${experiment_name}_33.log" &
pid4=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o $out_path -experiment 34 > "$log_path/${experiment_name}_34.log" &
pid5=$!

wait $pid1
wait $pid2
wait $pid3
wait $pid4
wait $pid5

