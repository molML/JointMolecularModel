#!/bin/bash
#SBATCH --job-name=vae_pretraining3_0_1_2_3_4
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/vae_pretraining3_0_1_2_3_4.out
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

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/vae_pretraining3/0 -experiment 0 > "$log_path/vae_pretraining3_0.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/vae_pretraining3/1 -experiment 1 > "$log_path/vae_pretraining3_1.log" &
pid2=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/vae_pretraining3/2 -experiment 2 > "$log_path/vae_pretraining3_2.log" &
pid3=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/vae_pretraining3/3 -experiment 3 > "$log_path/vae_pretraining3_3.log" &
pid4=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/vae_pretraining3/4 -experiment 4 > "$log_path/vae_pretraining3_4.log" &
pid5=$!

wait $pid1
wait $pid2
wait $pid3
wait $pid4
wait $pid5

cp -r $project_path/results/vae_pretraining3/0 /projects/prjs1021/JointChemicalModel/results/vae_pretraining3/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/vae_pretraining3/0
fi

cp -r $project_path/results/vae_pretraining3/1 /projects/prjs1021/JointChemicalModel/results/vae_pretraining3/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/vae_pretraining3/1
fi

cp -r $project_path/results/vae_pretraining3/2 /projects/prjs1021/JointChemicalModel/results/vae_pretraining3/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/vae_pretraining3/2
fi

cp -r $project_path/results/vae_pretraining3/3 /projects/prjs1021/JointChemicalModel/results/vae_pretraining3/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/vae_pretraining3/3
fi

cp -r $project_path/results/vae_pretraining3/4 /projects/prjs1021/JointChemicalModel/results/vae_pretraining3/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/vae_pretraining3/4
fi

