#!/bin/bash
#SBATCH --job-name=ae_pretraining_30_31
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/ae_pretraining_30_31.out
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=1
#SBATCH --time=120:00:00

project_path="$HOME/projects/JointChemicalModel"
experiment_script_path="$project_path/experiments/3.2_ae_pretraining.py"

log_path="$project_path/results/logs"

source $HOME/anaconda3/etc/profile.d/conda.sh
export PYTHONPATH="$PYTHONPATH:$project_path"

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ae_pretraining/30 -experiment 30 > "$log_path/ae_pretraining_30.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ae_pretraining/31 -experiment 31 > "$log_path/ae_pretraining_31.log" &
pid2=$!

wait $pid1
wait $pid2

cp -r $project_path/results/ae_pretraining/30 /projects/prjs1021/JointChemicalModel/results/ae_pretraining/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ae_pretraining/30
fi

cp -r $project_path/results/ae_pretraining/31 /projects/prjs1021/JointChemicalModel/results/ae_pretraining/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ae_pretraining/31
fi

