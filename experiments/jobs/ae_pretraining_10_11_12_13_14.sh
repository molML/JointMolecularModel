#!/bin/bash
#SBATCH --job-name=ae_pretraining_10_11_12_13_14
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/ae_pretraining_10_11_12_13_14.out
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=1
#SBATCH --time=120:00:00

project_path="$HOME/projects/JointChemicalModel"
experiment_script_path="$project_path/experiments/3.1_ae_pretraining.py"

log_path="$project_path/results/logs"

source $HOME/anaconda3/etc/profile.d/conda.sh
export PYTHONPATH="$PYTHONPATH:$project_path"

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ae_pretraining/10 -experiment 10 > "$log_path/ae_pretraining_10.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ae_pretraining/11 -experiment 11 > "$log_path/ae_pretraining_11.log" &
pid2=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ae_pretraining/12 -experiment 12 > "$log_path/ae_pretraining_12.log" &
pid3=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ae_pretraining/13 -experiment 13 > "$log_path/ae_pretraining_13.log" &
pid4=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ae_pretraining/14 -experiment 14 > "$log_path/ae_pretraining_14.log" &
pid5=$!

wait $pid1
wait $pid2
wait $pid3
wait $pid4
wait $pid5

cp -r $project_path/results/ae_pretraining/10 /projects/prjs1021/JointChemicalModel/results/ae_pretraining/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ae_pretraining/10
fi

cp -r $project_path/results/ae_pretraining/11 /projects/prjs1021/JointChemicalModel/results/ae_pretraining/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ae_pretraining/11
fi

cp -r $project_path/results/ae_pretraining/12 /projects/prjs1021/JointChemicalModel/results/ae_pretraining/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ae_pretraining/12
fi

cp -r $project_path/results/ae_pretraining/13 /projects/prjs1021/JointChemicalModel/results/ae_pretraining/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ae_pretraining/13
fi

cp -r $project_path/results/ae_pretraining/14 /projects/prjs1021/JointChemicalModel/results/ae_pretraining/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ae_pretraining/14
fi

