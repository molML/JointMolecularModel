#!/bin/bash
#SBATCH --job-name=vae_pretraining2_25_26_27_28_29
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/vae_pretraining2_25_26_27_28_29.out
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

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/vae_pretraining2/25 -experiment 25 > "$log_path/vae_pretraining2_25.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/vae_pretraining2/26 -experiment 26 > "$log_path/vae_pretraining2_26.log" &
pid2=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/vae_pretraining2/27 -experiment 27 > "$log_path/vae_pretraining2_27.log" &
pid3=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/vae_pretraining2/28 -experiment 28 > "$log_path/vae_pretraining2_28.log" &
pid4=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/vae_pretraining2/29 -experiment 29 > "$log_path/vae_pretraining2_29.log" &
pid5=$!

wait $pid1
wait $pid2
wait $pid3
wait $pid4
wait $pid5

cp -r $project_path/results/vae_pretraining2/25 /projects/prjs1021/JointChemicalModel/results/vae_pretraining2/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/vae_pretraining2/25
fi

cp -r $project_path/results/vae_pretraining2/26 /projects/prjs1021/JointChemicalModel/results/vae_pretraining2/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/vae_pretraining2/26
fi

cp -r $project_path/results/vae_pretraining2/27 /projects/prjs1021/JointChemicalModel/results/vae_pretraining2/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/vae_pretraining2/27
fi

cp -r $project_path/results/vae_pretraining2/28 /projects/prjs1021/JointChemicalModel/results/vae_pretraining2/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/vae_pretraining2/28
fi

cp -r $project_path/results/vae_pretraining2/29 /projects/prjs1021/JointChemicalModel/results/vae_pretraining2/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/vae_pretraining2/29
fi

