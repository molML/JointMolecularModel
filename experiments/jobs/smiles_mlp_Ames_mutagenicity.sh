#!/bin/bash
#SBATCH --job-name=smiles_mlp_CHEMBL235_EC50_CHEMBL2835_Ki_Ames_mutagenicity_CHEMBL237_EC50_CHEMBL236_Ki
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/smiles_mlp_Ames_mutagenicity.out
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

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/smiles_mlp/Ames_mutagenicity -dataset Ames_mutagenicity > "$log_path/smiles_mlp_Ames_mutagenicity.log"

cp -r $project_path/results/smiles_mlp/Ames_mutagenicity /projects/prjs1021/JointChemicalModel/results/smiles_mlp/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/smiles_mlp/Ames_mutagenicity
fi
