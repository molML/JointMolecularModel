#!/bin/bash
#SBATCH --job-name=vae_pretraining2_15_16_17_18_19
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/vae_pretraining2_15_16_17_18_19.out
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

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/vae_pretraining2/15 -experiment 15 > "$log_path/vae_pretraining2_15.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/vae_pretraining2/16 -experiment 16 > "$log_path/vae_pretraining2_16.log" &
pid2=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/vae_pretraining2/17 -experiment 17 > "$log_path/vae_pretraining2_17.log" &
pid3=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/vae_pretraining2/18 -experiment 18 > "$log_path/vae_pretraining2_18.log" &
pid4=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/vae_pretraining2/19 -experiment 19 > "$log_path/vae_pretraining2_19.log" &
pid5=$!

wait $pid1
wait $pid2
wait $pid3
wait $pid4
wait $pid5

cp -r $project_path/results/vae_pretraining2/15 /projects/prjs1021/JointChemicalModel/results/vae_pretraining2/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/vae_pretraining2/15
fi

cp -r $project_path/results/vae_pretraining2/16 /projects/prjs1021/JointChemicalModel/results/vae_pretraining2/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/vae_pretraining2/16
fi

cp -r $project_path/results/vae_pretraining2/17 /projects/prjs1021/JointChemicalModel/results/vae_pretraining2/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/vae_pretraining2/17
fi

cp -r $project_path/results/vae_pretraining2/18 /projects/prjs1021/JointChemicalModel/results/vae_pretraining2/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/vae_pretraining2/18
fi

cp -r $project_path/results/vae_pretraining2/19 /projects/prjs1021/JointChemicalModel/results/vae_pretraining2/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/vae_pretraining2/19
fi

