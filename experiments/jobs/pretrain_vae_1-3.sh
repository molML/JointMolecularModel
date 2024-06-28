#!/bin/bash
#SBATCH --job-name=vae_pretraining_1_3
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/pretrain_vae_1_3.out
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00

experiment_name="vae_pretraining"

echo 'experiment_name = $experiment_name'

project_path="$HOME/projects/JointChemicalModel"
experiment_script_path="$project_path/experiments/3.0_vae_pretraining.py"

out_path="$project_path/results/$experiment_name"
log_path="$HOME/results/logs"

echo 'past the vars'


source $HOME/anaconda3/etc/profile.d/conda.sh
export PYTHONPATH="$PYTHONPATH:$project_path"



$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o $out_path -experiment 1 > "$log_path/${experiment_name}_1.log" &
pid1=$!

echo 'past job 1'

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o $out_path -experiment 2 > "$log_path/${experiment_name}_2.log" &
pid2=$!

echo 'past job 2'

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o $out_path -experiment 3 > "$log_path/${experiment_name}_3.log" &
pid3=$!

echo 'past job 3'

wait $pid1
wait $pid2
wait $pid3
