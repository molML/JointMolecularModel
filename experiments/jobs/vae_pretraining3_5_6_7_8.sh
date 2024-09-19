#!/bin/bash
#SBATCH --job-name=vae_pretraining3_5_6_7_8
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/vae_pretraining3_5_6_7_8.out
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=1
#SBATCH --time=120:00:00

project_path="$HOME/projects/JointChemicalModel"
experiment_script_path="$project_path/experiments/3.0.2_vae_pretraining3.py"

log_path="$project_path/results/logs"

source $HOME/anaconda3/etc/profile.d/conda.sh
export PYTHONPATH="$PYTHONPATH:$project_path"

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/vae_pretraining3/5 -experiment 5 > "$log_path/vae_pretraining3_5.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/vae_pretraining3/6 -experiment 6 > "$log_path/vae_pretraining3_6.log" &
pid2=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/vae_pretraining3/7 -experiment 7 > "$log_path/vae_pretraining3_7.log" &
pid3=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/vae_pretraining3/8 -experiment 8 > "$log_path/vae_pretraining3_8.log" &
pid4=$!

wait $pid1
wait $pid2
wait $pid3
wait $pid4

cp -r $project_path/results/vae_pretraining3/5 /projects/prjs1021/JointChemicalModel/results/vae_pretraining3/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/vae_pretraining3/5
fi

cp -r $project_path/results/vae_pretraining3/6 /projects/prjs1021/JointChemicalModel/results/vae_pretraining3/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/vae_pretraining3/6
fi

cp -r $project_path/results/vae_pretraining3/7 /projects/prjs1021/JointChemicalModel/results/vae_pretraining3/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/vae_pretraining3/7
fi

cp -r $project_path/results/vae_pretraining3/8 /projects/prjs1021/JointChemicalModel/results/vae_pretraining3/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/vae_pretraining3/8
fi

