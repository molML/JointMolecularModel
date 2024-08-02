#!/bin/bash
#SBATCH --job-name=ecfp_mlp_10_11_12_13_14
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/ecfp_mlp_10_11_12_13_14.out
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

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ecfp_mlp/CHEMBL4792_Ki -experiment 10 > "$log_path/ecfp_mlp_10.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ecfp_mlp/CHEMBL4203_Ki -experiment 11 > "$log_path/ecfp_mlp_11.log" &
pid2=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ecfp_mlp/CHEMBL219_Ki -experiment 12 > "$log_path/ecfp_mlp_12.log" &
pid3=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ecfp_mlp/CHEMBL228_Ki -experiment 13 > "$log_path/ecfp_mlp_13.log" &
pid4=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ecfp_mlp/CHEMBL2034_Ki -experiment 14 > "$log_path/ecfp_mlp_14.log" &
pid5=$!

wait $pid1
wait $pid2
wait $pid3
wait $pid4
wait $pid5

cp $project_path/results/ecfp_mlp/CHEMBL4792_Ki /projects/prjs1021/JointChemicalModel/results/ecfp_mlp/CHEMBL4792_Ki
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ecfp_mlp/CHEMBL4792_Ki
fi

cp $project_path/results/ecfp_mlp/CHEMBL4203_Ki /projects/prjs1021/JointChemicalModel/results/ecfp_mlp/CHEMBL4203_Ki
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ecfp_mlp/CHEMBL4203_Ki
fi

cp $project_path/results/ecfp_mlp/CHEMBL219_Ki /projects/prjs1021/JointChemicalModel/results/ecfp_mlp/CHEMBL219_Ki
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ecfp_mlp/CHEMBL219_Ki
fi

cp $project_path/results/ecfp_mlp/CHEMBL228_Ki /projects/prjs1021/JointChemicalModel/results/ecfp_mlp/CHEMBL228_Ki
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ecfp_mlp/CHEMBL228_Ki
fi

cp $project_path/results/ecfp_mlp/CHEMBL2034_Ki /projects/prjs1021/JointChemicalModel/results/ecfp_mlp/CHEMBL2034_Ki
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ecfp_mlp/CHEMBL2034_Ki
fi

