#!/bin/bash
#SBATCH --job-name=vae_pretraining2_95
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/vae_pretraining2_95.out
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

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/vae_pretraining2/95 -experiment 95 > "$log_path/vae_pretraining2_95.log" &
pid1=$!

wait $pid1

cp -r $project_path/results/vae_pretraining2/95 /projects/prjs1021/JointChemicalModel/results/vae_pretraining2/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/vae_pretraining2/95
fi

