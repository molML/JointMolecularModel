#!/bin/bash
#SBATCH --job-name=ecfp_mlp_fix1
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/ecfp_mlp_fix1.out
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=1
#SBATCH --time=36:00:00

project_path="$HOME/projects/JointChemicalModel"
experiment_script_path="$project_path/experiments/4.2_ecfp_mlp.py"

log_path="$project_path/results/logs"

source $HOME/anaconda3/etc/profile.d/conda.sh
export PYTHONPATH="$PYTHONPATH:$project_path"

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ecfp_mlp/CHEMBL236_Ki -experiment 19 > "$log_path/ecfp_mlp_19.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ecfp_mlp/CHEMBL237_EC50 -experiment 18 > "$log_path/ecfp_mlp_18.log" &
pid2=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ecfp_mlp/CHEMBL4203_Ki -experiment 11 > "$log_path/ecfp_mlp_11.log" &
pid3=$!

wait $pid1
wait $pid2
wait $pid3

cp -r $project_path/results/ecfp_mlp/CHEMBL236_Ki /projects/prjs1021/JointChemicalModel/results/ecfp_mlp/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ecfp_mlp/CHEMBL236_Ki
fi

cp -r $project_path/results/ecfp_mlp/CHEMBL237_EC50 /projects/prjs1021/JointChemicalModel/results/ecfp_mlp/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ecfp_mlp/CHEMBL237_EC50
fi

cp -r $project_path/results/ecfp_mlp/CHEMBL4203_Ki /projects/prjs1021/JointChemicalModel/results/ecfp_mlp/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ecfp_mlp/CHEMBL4203_Ki
fi

