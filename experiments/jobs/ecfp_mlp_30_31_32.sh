#!/bin/bash
#SBATCH --job-name=ecfp_mlp_30_31_32
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/ecfp_mlp_30_31_32.out
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00

project_path="$HOME/projects/JointChemicalModel"
experiment_script_path="$project_path/experiments/4.2_ecfp_mlp.py"

out_path="$project_path/results"
log_path="$project_path/results/logs"

source $HOME/anaconda3/etc/profile.d/conda.sh
export PYTHONPATH="$PYTHONPATH:$project_path"

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o $out_path -experiment 30 > "$log_path/${experiment_name}_30.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o $out_path -experiment 31 > "$log_path/${experiment_name}_31.log" &
pid2=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o $out_path -experiment 32 > "$log_path/${experiment_name}_32.log" &
pid3=$!

wait $pid1
wait $pid2
wait $pid3

