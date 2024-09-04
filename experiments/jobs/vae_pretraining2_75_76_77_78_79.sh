#!/bin/bash
#SBATCH --job-name=vae_pretraining2_75_76_77_78_79
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/vae_pretraining2_75_76_77_78_79.out
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

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/vae_pretraining2/75 -experiment 75 > "$log_path/vae_pretraining2_75.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/vae_pretraining2/76 -experiment 76 > "$log_path/vae_pretraining2_76.log" &
pid2=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/vae_pretraining2/77 -experiment 77 > "$log_path/vae_pretraining2_77.log" &
pid3=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/vae_pretraining2/78 -experiment 78 > "$log_path/vae_pretraining2_78.log" &
pid4=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/vae_pretraining2/79 -experiment 79 > "$log_path/vae_pretraining2_79.log" &
pid5=$!

wait $pid1
wait $pid2
wait $pid3
wait $pid4
wait $pid5

cp -r $project_path/results/vae_pretraining2/75 /projects/prjs1021/JointChemicalModel/results/vae_pretraining2/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/vae_pretraining2/75
fi

cp -r $project_path/results/vae_pretraining2/76 /projects/prjs1021/JointChemicalModel/results/vae_pretraining2/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/vae_pretraining2/76
fi

cp -r $project_path/results/vae_pretraining2/77 /projects/prjs1021/JointChemicalModel/results/vae_pretraining2/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/vae_pretraining2/77
fi

cp -r $project_path/results/vae_pretraining2/78 /projects/prjs1021/JointChemicalModel/results/vae_pretraining2/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/vae_pretraining2/78
fi

cp -r $project_path/results/vae_pretraining2/79 /projects/prjs1021/JointChemicalModel/results/vae_pretraining2/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/vae_pretraining2/79
fi

