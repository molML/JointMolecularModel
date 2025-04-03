#!/bin/bash
#SBATCH --job-name=ae_pretraining_15_16_17_18_19
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/ae_pretraining_15_16_17_18_19.out
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

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ae_pretraining/15 -experiment 15 > "$log_path/ae_pretraining_15.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ae_pretraining/16 -experiment 16 > "$log_path/ae_pretraining_16.log" &
pid2=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ae_pretraining/17 -experiment 17 > "$log_path/ae_pretraining_17.log" &
pid3=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ae_pretraining/18 -experiment 18 > "$log_path/ae_pretraining_18.log" &
pid4=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ae_pretraining/19 -experiment 19 > "$log_path/ae_pretraining_19.log" &
pid5=$!

wait $pid1
wait $pid2
wait $pid3
wait $pid4
wait $pid5

cp -r $project_path/results/ae_pretraining/15 /projects/prjs1021/JointChemicalModel/results/ae_pretraining/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ae_pretraining/15
fi

cp -r $project_path/results/ae_pretraining/16 /projects/prjs1021/JointChemicalModel/results/ae_pretraining/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ae_pretraining/16
fi

cp -r $project_path/results/ae_pretraining/17 /projects/prjs1021/JointChemicalModel/results/ae_pretraining/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ae_pretraining/17
fi

cp -r $project_path/results/ae_pretraining/18 /projects/prjs1021/JointChemicalModel/results/ae_pretraining/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ae_pretraining/18
fi

cp -r $project_path/results/ae_pretraining/19 /projects/prjs1021/JointChemicalModel/results/ae_pretraining/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ae_pretraining/19
fi

