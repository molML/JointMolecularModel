#!/bin/bash
#SBATCH --job-name=cats_mlp_15_16_17_18_19
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/cats_mlp_15_16_17_18_19.out
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

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/cats_mlp/CHEMBL235_EC50 -experiment cats_mlp_15 > "$log_path/cats_mlp_15.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/cats_mlp/CHEMBL2835_Ki -experiment cats_mlp_16 > "$log_path/cats_mlp_16.log" &
pid2=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/cats_mlp/Ames_mutagenicity -experiment cats_mlp_17 > "$log_path/cats_mlp_17.log" &
pid3=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/cats_mlp/CHEMBL237_EC50 -experiment cats_mlp_18 > "$log_path/cats_mlp_18.log" &
pid4=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/cats_mlp/CHEMBL236_Ki -experiment cats_mlp_19 > "$log_path/cats_mlp_19.log" &
pid5=$!

wait $pid1
wait $pid2
wait $pid3
wait $pid4
wait $pid5

mv $HOME/results/cats_mlp/CHEMBL235_EC50 $HOME/../../projects/prjs1021/JointChemicalModel/results/cats_mlp

mv $HOME/results/cats_mlp/CHEMBL2835_Ki $HOME/../../projects/prjs1021/JointChemicalModel/results/cats_mlp

mv $HOME/results/cats_mlp/Ames_mutagenicity $HOME/../../projects/prjs1021/JointChemicalModel/results/cats_mlp

mv $HOME/results/cats_mlp/CHEMBL237_EC50 $HOME/../../projects/prjs1021/JointChemicalModel/results/cats_mlp

mv $HOME/results/cats_mlp/CHEMBL236_Ki $HOME/../../projects/prjs1021/JointChemicalModel/results/cats_mlp


