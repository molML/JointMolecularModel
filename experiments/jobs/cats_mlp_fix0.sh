#!/bin/bash
#SBATCH --job-name=cats_mlp_fix0
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/cats_mlp_fix0.out
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=1
#SBATCH --time=42:00:00

project_path="$HOME/projects/JointChemicalModel"
experiment_script_path="$project_path/experiments/4.3_cats_mlp.py"

log_path="$project_path/results/logs"

source $HOME/anaconda3/etc/profile.d/conda.sh
export PYTHONPATH="$PYTHONPATH:$project_path"

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/cats_mlp/CHEMBL231_Ki -experiment 20 > "$log_path/cats_mlp_20.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/cats_mlp/CHEMBL233_Ki -experiment 4 > "$log_path/cats_mlp_4.log" &
pid2=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/cats_mlp/CHEMBL234_Ki -experiment 5 > "$log_path/cats_mlp_5.log" &
pid3=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/cats_mlp/CHEMBL2971_Ki -experiment 6 > "$log_path/cats_mlp_6.log" &
pid4=$!

wait $pid1
wait $pid2
wait $pid3
wait $pid4

cp -r $project_path/results/cats_mlp/CHEMBL231_Ki /projects/prjs1021/JointChemicalModel/results/cats_mlp/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/cats_mlp/CHEMBL231_Ki
fi

cp -r $project_path/results/cats_mlp/CHEMBL233_Ki /projects/prjs1021/JointChemicalModel/results/cats_mlp/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/cats_mlp/CHEMBL233_Ki
fi

cp -r $project_path/results/cats_mlp/CHEMBL234_Ki /projects/prjs1021/JointChemicalModel/results/cats_mlp/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/cats_mlp/CHEMBL234_Ki
fi

cp -r $project_path/results/cats_mlp/CHEMBL2971_Ki /projects/prjs1021/JointChemicalModel/results/cats_mlp/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/cats_mlp/CHEMBL2971_Ki
fi

