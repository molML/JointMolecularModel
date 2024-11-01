#!/bin/bash
#SBATCH --job-name=smiles_var_mlp_CHEMBL231_Ki_CHEMBL264_Ki_ESR1_ant_CHEMBL1871_Ki_TP53
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/smiles_var_mlp_CHEMBL231_Ki_CHEMBL264_Ki_ESR1_ant_CHEMBL1871_Ki_TP53.out
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=1
#SBATCH --time=80:00:00

project_path="$HOME/projects/JointChemicalModel"
experiment_script_path="$project_path/experiments/4.4_smiles_var_mlp.py"

log_path="$project_path/results/logs"

source $HOME/anaconda3/etc/profile.d/conda.sh
export PYTHONPATH="$PYTHONPATH:$project_path"

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/smiles_var_mlp/CHEMBL231_Ki -dataset CHEMBL231_Ki > "$log_path/smiles_var_mlp_CHEMBL231_Ki.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/smiles_var_mlp/CHEMBL264_Ki -dataset CHEMBL264_Ki > "$log_path/smiles_var_mlp_CHEMBL264_Ki.log" &
pid2=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/smiles_var_mlp/ESR1_ant -dataset ESR1_ant > "$log_path/smiles_var_mlp_ESR1_ant.log" &
pid3=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/smiles_var_mlp/CHEMBL1871_Ki -dataset CHEMBL1871_Ki > "$log_path/smiles_var_mlp_CHEMBL1871_Ki.log" &
pid4=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/smiles_var_mlp/TP53 -dataset TP53 > "$log_path/smiles_mlp_TP53.log" &
pid5=$!

wait $pid1
wait $pid2
wait $pid3
wait $pid4
wait $pid5

cp -r $project_path/results/smiles_mlp/CHEMBL231_Ki /projects/prjs1021/JointChemicalModel/results/smiles_mlp/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/smiles_mlp/CHEMBL231_Ki
fi

cp -r $project_path/results/smiles_mlp/CHEMBL264_Ki /projects/prjs1021/JointChemicalModel/results/smiles_mlp/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/smiles_mlp/CHEMBL264_Ki
fi

cp -r $project_path/results/smiles_mlp/ESR1_ant /projects/prjs1021/JointChemicalModel/results/smiles_mlp/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/smiles_mlp/ESR1_ant
fi

cp -r $project_path/results/smiles_mlp/CHEMBL1871_Ki /projects/prjs1021/JointChemicalModel/results/smiles_mlp/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/smiles_mlp/CHEMBL1871_Ki
fi

cp -r $project_path/results/smiles_mlp/TP53 /projects/prjs1021/JointChemicalModel/results/smiles_mlp/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/smiles_mlp/TP53
fi

