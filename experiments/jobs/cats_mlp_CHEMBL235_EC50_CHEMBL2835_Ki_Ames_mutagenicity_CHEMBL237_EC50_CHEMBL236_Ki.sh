#!/bin/bash
#SBATCH --job-name=cats_mlp_CHEMBL235_EC50_CHEMBL2835_Ki_Ames_mutagenicity_CHEMBL237_EC50_CHEMBL236_Ki
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/cats_mlp_CHEMBL235_EC50_CHEMBL2835_Ki_Ames_mutagenicity_CHEMBL237_EC50_CHEMBL236_Ki.out
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

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/cats_mlp/CHEMBL235_EC50 -dataset CHEMBL235_EC50 > "$log_path/cats_mlp_CHEMBL235_EC50.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/cats_mlp/CHEMBL2835_Ki -dataset CHEMBL2835_Ki > "$log_path/cats_mlp_CHEMBL2835_Ki.log" &
pid2=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/cats_mlp/Ames_mutagenicity -dataset Ames_mutagenicity > "$log_path/cats_mlp_Ames_mutagenicity.log" &
pid3=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/cats_mlp/CHEMBL237_EC50 -dataset CHEMBL237_EC50 > "$log_path/cats_mlp_CHEMBL237_EC50.log" &
pid4=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/cats_mlp/CHEMBL236_Ki -dataset CHEMBL236_Ki > "$log_path/cats_mlp_CHEMBL236_Ki.log" &
pid5=$!

wait $pid1
wait $pid2
wait $pid3
wait $pid4
wait $pid5

cp -r $project_path/results/cats_mlp/CHEMBL235_EC50 /projects/prjs1021/JointChemicalModel/results/cats_mlp/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/cats_mlp/CHEMBL235_EC50
fi

cp -r $project_path/results/cats_mlp/CHEMBL2835_Ki /projects/prjs1021/JointChemicalModel/results/cats_mlp/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/cats_mlp/CHEMBL2835_Ki
fi

cp -r $project_path/results/cats_mlp/Ames_mutagenicity /projects/prjs1021/JointChemicalModel/results/cats_mlp/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/cats_mlp/Ames_mutagenicity
fi

cp -r $project_path/results/cats_mlp/CHEMBL237_EC50 /projects/prjs1021/JointChemicalModel/results/cats_mlp/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/cats_mlp/CHEMBL237_EC50
fi

cp -r $project_path/results/cats_mlp/CHEMBL236_Ki /projects/prjs1021/JointChemicalModel/results/cats_mlp/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/cats_mlp/CHEMBL236_Ki
fi

