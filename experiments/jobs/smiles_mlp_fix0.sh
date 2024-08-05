#!/bin/bash
#SBATCH --job-name=smiles_mlp_fix0
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/smiles_mlp_fix0.out
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=1
#SBATCH --time=80:00:00

project_path="$HOME/projects/JointChemicalModel"
experiment_script_path="$project_path/experiments/4.4_smiles_mlp.py"

log_path="$project_path/results/logs"

source $HOME/anaconda3/etc/profile.d/conda.sh
export PYTHONPATH="$PYTHONPATH:$project_path"


$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/smiles_mlp/CHEMBL2835_Ki -experiment 16 > "$log_path/smiles_mlp_16.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/smiles_mlp/CHEMBL287_Ki -experiment 8 > "$log_path/smiles_mlp_8.log" &
pid2=$!

wait $pid1
wait $pid2

cp -r $project_path/results/smiles_mlp/CHEMBL2835_Ki /projects/prjs1021/JointChemicalModel/results/smiles_mlp/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/smiles_mlp/CHEMBL2835_Ki
fi

cp -r $project_path/results/smiles_mlp/CHEMBL287_Ki /projects/prjs1021/JointChemicalModel/results/smiles_mlp/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/smiles_mlp/CHEMBL287_Ki
fi

