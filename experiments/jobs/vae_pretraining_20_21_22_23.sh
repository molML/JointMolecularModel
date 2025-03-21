#!/bin/bash
#SBATCH --job-name=vae_pretraining_20_21_22_23
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/vae_pretraining_20_21_22_23.out
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=1
#SBATCH --time=120:00:00

project_path="$HOME/projects/JointChemicalModel"
experiment_script_path="$project_path/experiments/3.0_vae_pretraining.py"

log_path="$project_path/results/logs"

source $HOME/anaconda3/etc/profile.d/conda.sh
export PYTHONPATH="$PYTHONPATH:$project_path"

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/vae_pretraining/20 -experiment 20 > "$log_path/vae_pretraining_20.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/vae_pretraining/21 -experiment 21 > "$log_path/vae_pretraining_21.log" &
pid2=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/vae_pretraining/22 -experiment 22 > "$log_path/vae_pretraining_22.log" &
pid3=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/vae_pretraining/23 -experiment 23 > "$log_path/vae_pretraining_23.log" &
pid4=$!

wait $pid1
wait $pid2
wait $pid3
wait $pid4

cp -r $project_path/results/vae_pretraining/20 /projects/prjs1021/JointChemicalModel/results/vae_pretraining/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/vae_pretraining/20
fi

cp -r $project_path/results/vae_pretraining/21 /projects/prjs1021/JointChemicalModel/results/vae_pretraining/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/vae_pretraining/21
fi

cp -r $project_path/results/vae_pretraining/22 /projects/prjs1021/JointChemicalModel/results/vae_pretraining/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/vae_pretraining/22
fi

cp -r $project_path/results/vae_pretraining/23 /projects/prjs1021/JointChemicalModel/results/vae_pretraining/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/vae_pretraining/23
fi

