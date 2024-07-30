#!/bin/bash
#SBATCH --job-name=cats_mlp_25_26_27_28_29
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/cats_mlp_25_26_27_28_29.out
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

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/cats_mlp/CHEMBL3979_EC50 -experiment cats_mlp_25 > "$log_path/cats_mlp_25.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/cats_mlp/CHEMBL4005_Ki -experiment cats_mlp_26 > "$log_path/cats_mlp_26.log" &
pid2=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/cats_mlp/CHEMBL4616_EC50 -experiment cats_mlp_27 > "$log_path/cats_mlp_27.log" &
pid3=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/cats_mlp/CHEMBL262_Ki -experiment cats_mlp_28 > "$log_path/cats_mlp_28.log" &
pid4=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/cats_mlp/CHEMBL237_Ki -experiment cats_mlp_29 > "$log_path/cats_mlp_29.log" &
pid5=$!

wait $pid1
wait $pid2
wait $pid3
wait $pid4
wait $pid5

mv $HOME/results/cats_mlp/CHEMBL3979_EC50 $HOME/../../projects/prjs1021/JointChemicalModel/results/cats_mlp

mv $HOME/results/cats_mlp/CHEMBL4005_Ki $HOME/../../projects/prjs1021/JointChemicalModel/results/cats_mlp

mv $HOME/results/cats_mlp/CHEMBL4616_EC50 $HOME/../../projects/prjs1021/JointChemicalModel/results/cats_mlp

mv $HOME/results/cats_mlp/CHEMBL262_Ki $HOME/../../projects/prjs1021/JointChemicalModel/results/cats_mlp

mv $HOME/results/cats_mlp/CHEMBL237_Ki $HOME/../../projects/prjs1021/JointChemicalModel/results/cats_mlp


