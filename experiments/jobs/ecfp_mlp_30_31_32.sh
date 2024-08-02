#!/bin/bash
#SBATCH --job-name=ecfp_mlp_30_31_32
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/ecfp_mlp_30_31_32.out
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

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ecfp_mlp/CHEMBL1862_Ki -experiment 30 > "$log_path/ecfp_mlp_30.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ecfp_mlp/CHEMBL214_Ki -experiment 31 > "$log_path/ecfp_mlp_31.log" &
pid2=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ecfp_mlp/CHEMBL2047_EC50 -experiment 32 > "$log_path/ecfp_mlp_32.log" &
pid3=$!

wait $pid1
wait $pid2
wait $pid3

cp $project_path/results/ecfp_mlp/CHEMBL1862_Ki /projects/prjs1021/JointChemicalModel/results/ecfp_mlp/CHEMBL1862_Ki
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ecfp_mlp/CHEMBL1862_Ki
fi

cp $project_path/results/ecfp_mlp/CHEMBL214_Ki /projects/prjs1021/JointChemicalModel/results/ecfp_mlp/CHEMBL214_Ki
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ecfp_mlp/CHEMBL214_Ki
fi

cp $project_path/results/ecfp_mlp/CHEMBL2047_EC50 /projects/prjs1021/JointChemicalModel/results/ecfp_mlp/CHEMBL2047_EC50
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ecfp_mlp/CHEMBL2047_EC50
fi

