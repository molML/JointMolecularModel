#!/bin/bash
#SBATCH --job-name=vae_pretraining2_35_36_37_38_39
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/vae_pretraining2_35_36_37_38_39.out
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

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/vae_pretraining2/35 -experiment 35 > "$log_path/vae_pretraining2_35.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/vae_pretraining2/36 -experiment 36 > "$log_path/vae_pretraining2_36.log" &
pid2=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/vae_pretraining2/37 -experiment 37 > "$log_path/vae_pretraining2_37.log" &
pid3=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/vae_pretraining2/38 -experiment 38 > "$log_path/vae_pretraining2_38.log" &
pid4=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/vae_pretraining2/39 -experiment 39 > "$log_path/vae_pretraining2_39.log" &
pid5=$!

wait $pid1
wait $pid2
wait $pid3
wait $pid4
wait $pid5

cp -r $project_path/results/vae_pretraining2/35 /projects/prjs1021/JointChemicalModel/results/vae_pretraining2/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/vae_pretraining2/35
fi

cp -r $project_path/results/vae_pretraining2/36 /projects/prjs1021/JointChemicalModel/results/vae_pretraining2/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/vae_pretraining2/36
fi

cp -r $project_path/results/vae_pretraining2/37 /projects/prjs1021/JointChemicalModel/results/vae_pretraining2/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/vae_pretraining2/37
fi

cp -r $project_path/results/vae_pretraining2/38 /projects/prjs1021/JointChemicalModel/results/vae_pretraining2/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/vae_pretraining2/38
fi

cp -r $project_path/results/vae_pretraining2/39 /projects/prjs1021/JointChemicalModel/results/vae_pretraining2/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/vae_pretraining2/39
fi

