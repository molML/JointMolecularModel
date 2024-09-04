#!/bin/bash
#SBATCH --job-name=vae_pretraining2_80_81_82_83_84
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/vae_pretraining2_80_81_82_83_84.out
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=1
#SBATCH --time=120:00:00

project_path="$HOME/projects/JointChemicalModel"
experiment_script_path="$project_path/experiments/3.0.1_vae_pretraining2.py"

log_path="$project_path/results/logs"

source $HOME/anaconda3/etc/profile.d/conda.sh
export PYTHONPATH="$PYTHONPATH:$project_path"

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/vae_pretraining2/80 -experiment 80 > "$log_path/vae_pretraining2_80.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/vae_pretraining2/81 -experiment 81 > "$log_path/vae_pretraining2_81.log" &
pid2=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/vae_pretraining2/82 -experiment 82 > "$log_path/vae_pretraining2_82.log" &
pid3=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/vae_pretraining2/83 -experiment 83 > "$log_path/vae_pretraining2_83.log" &
pid4=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/vae_pretraining2/84 -experiment 84 > "$log_path/vae_pretraining2_84.log" &
pid5=$!

wait $pid1
wait $pid2
wait $pid3
wait $pid4
wait $pid5

cp -r $project_path/results/vae_pretraining2/80 /projects/prjs1021/JointChemicalModel/results/vae_pretraining2/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/vae_pretraining2/80
fi

cp -r $project_path/results/vae_pretraining2/81 /projects/prjs1021/JointChemicalModel/results/vae_pretraining2/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/vae_pretraining2/81
fi

cp -r $project_path/results/vae_pretraining2/82 /projects/prjs1021/JointChemicalModel/results/vae_pretraining2/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/vae_pretraining2/82
fi

cp -r $project_path/results/vae_pretraining2/83 /projects/prjs1021/JointChemicalModel/results/vae_pretraining2/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/vae_pretraining2/83
fi

cp -r $project_path/results/vae_pretraining2/84 /projects/prjs1021/JointChemicalModel/results/vae_pretraining2/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/vae_pretraining2/84
fi

