#!/bin/bash
#SBATCH --job-name=ecfp_mlp_15_16_17_18_19
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/ecfp_mlp_15_16_17_18_19.out
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00

project_path="$HOME/projects/JointChemicalModel"
experiment_script_path="$project_path/experiments/4.2_ecfp_mlp.py"

log_path="$project_path/results/logs"

source $HOME/anaconda3/etc/profile.d/conda.sh
export PYTHONPATH="$PYTHONPATH:$project_path"

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ecfp_mlp/CHEMBL235_EC50 -experiment 15 > "$log_path/ecfp_mlp_15.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ecfp_mlp/CHEMBL2835_Ki -experiment 16 > "$log_path/ecfp_mlp_16.log" &
pid2=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ecfp_mlp/Ames_mutagenicity -experiment 17 > "$log_path/ecfp_mlp_17.log" &
pid3=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ecfp_mlp/CHEMBL237_EC50 -experiment 18 > "$log_path/ecfp_mlp_18.log" &
pid4=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ecfp_mlp/CHEMBL236_Ki -experiment 19 > "$log_path/ecfp_mlp_19.log" &
pid5=$!

wait $pid1
wait $pid2
wait $pid3
wait $pid4
wait $pid5

cp $project_path/results/ecfp_mlp/CHEMBL235_EC50 /projects/prjs1021/JointChemicalModel/results/ecfp_mlp/CHEMBL235_EC50
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ecfp_mlp/CHEMBL235_EC50
fi

cp $project_path/results/ecfp_mlp/CHEMBL2835_Ki /projects/prjs1021/JointChemicalModel/results/ecfp_mlp/CHEMBL2835_Ki
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ecfp_mlp/CHEMBL2835_Ki
fi

cp $project_path/results/ecfp_mlp/Ames_mutagenicity /projects/prjs1021/JointChemicalModel/results/ecfp_mlp/Ames_mutagenicity
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ecfp_mlp/Ames_mutagenicity
fi

cp $project_path/results/ecfp_mlp/CHEMBL237_EC50 /projects/prjs1021/JointChemicalModel/results/ecfp_mlp/CHEMBL237_EC50
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ecfp_mlp/CHEMBL237_EC50
fi

cp $project_path/results/ecfp_mlp/CHEMBL236_Ki /projects/prjs1021/JointChemicalModel/results/ecfp_mlp/CHEMBL236_Ki
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ecfp_mlp/CHEMBL236_Ki
fi

