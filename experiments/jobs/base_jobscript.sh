#!/bin/bash
#SBATCH --job-name=JOBNAME
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/JOBNAME.out
#SBATCH -p PARTITION
#SBATCH -N 1
#SBATCH --ntasks=NTASKS
#SBATCH --gpus-per-node=GPUS_PER_NODE
#SBATCH --time=TIME

experiment_name=EXPERIMENT_NAME

project_path="$HOME/projects/JointChemicalModel"
experiment_script_path="$project_path/experiments/EXPERIMENT_SCRIPT"

out_path="$project_path/results/$experiment_name"
log_path="$project_path/results/logs"

source $HOME/anaconda3/etc/profile.d/conda.sh
export PYTHONPATH="$PYTHONPATH:$project_path"

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o $out_path -experiment EX1 > "$log_path/${experiment_name}_EX1.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o $out_path -experiment EX2 > "$log_path/${experiment_name}_EX2.log" &
pid2=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o $out_path -experiment EX3 > "$log_path/${experiment_name}_EX3.log" &
pid3=$!

wait $pid1
wait $pid2
wait $pid3
