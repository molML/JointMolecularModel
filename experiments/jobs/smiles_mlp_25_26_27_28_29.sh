#!/bin/bash
#SBATCH --job-name=smiles_mlp_25_26_27_28_29
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/smiles_mlp_25_26_27_28_29.out
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=1
#SBATCH --time=120:00:00

project_path="$HOME/projects/JointChemicalModel"
experiment_script_path="$project_path/experiments/4.4_smiles_mlp.py"

log_path="$project_path/results/logs"

source $HOME/anaconda3/etc/profile.d/conda.sh
export PYTHONPATH="$PYTHONPATH:$project_path"

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/smiles_mlp/CHEMBL3979_EC50 -experiment 25 > "$log_path/smiles_mlp_25.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/smiles_mlp/CHEMBL4005_Ki -experiment 26 > "$log_path/smiles_mlp_26.log" &
pid2=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/smiles_mlp/CHEMBL4616_EC50 -experiment 27 > "$log_path/smiles_mlp_27.log" &
pid3=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/smiles_mlp/CHEMBL262_Ki -experiment 28 > "$log_path/smiles_mlp_28.log" &
pid4=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/smiles_mlp/CHEMBL237_Ki -experiment 29 > "$log_path/smiles_mlp_29.log" &
pid5=$!

wait $pid1
wait $pid2
wait $pid3
wait $pid4
wait $pid5

mv $project_path/results/smiles_mlp/CHEMBL3979_EC50 /projects/prjs1021/JointChemicalModel/results/smiles_mlp

mv $project_path/results/smiles_mlp/CHEMBL4005_Ki /projects/prjs1021/JointChemicalModel/results/smiles_mlp

mv $project_path/results/smiles_mlp/CHEMBL4616_EC50 /projects/prjs1021/JointChemicalModel/results/smiles_mlp

mv $project_path/results/smiles_mlp/CHEMBL262_Ki /projects/prjs1021/JointChemicalModel/results/smiles_mlp

mv $project_path/results/smiles_mlp/CHEMBL237_Ki /projects/prjs1021/JointChemicalModel/results/smiles_mlp


