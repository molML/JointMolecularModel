#!/bin/bash
#SBATCH --job-name=cats_mlp_30_31_32
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/cats_mlp_30_31_32.out
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=1
#SBATCH --time=36:00:00

project_path="$HOME/projects/JointChemicalModel"
experiment_script_path="$project_path/experiments/4.3_cats_mlp.py"

log_path="$project_path/results/logs"

source $HOME/anaconda3/etc/profile.d/conda.sh
export PYTHONPATH="$PYTHONPATH:$project_path"

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/cats_mlp/CHEMBL1862_Ki -experiment cats_mlp_30 > "$log_path/cats_mlp_30.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/cats_mlp/CHEMBL214_Ki -experiment cats_mlp_31 > "$log_path/cats_mlp_31.log" &
pid2=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/cats_mlp/CHEMBL2047_EC50 -experiment cats_mlp_32 > "$log_path/cats_mlp_32.log" &
pid3=$!

wait $pid1
wait $pid2
wait $pid3

mv $HOME/results/cats_mlp/CHEMBL1862_Ki $HOME/../../projects/prjs1021/JointChemicalModel/results/cats_mlp

mv $HOME/results/cats_mlp/CHEMBL214_Ki $HOME/../../projects/prjs1021/JointChemicalModel/results/cats_mlp

mv $HOME/results/cats_mlp/CHEMBL2047_EC50 $HOME/../../projects/prjs1021/JointChemicalModel/results/cats_mlp


