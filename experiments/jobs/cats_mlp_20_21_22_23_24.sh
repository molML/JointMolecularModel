#!/bin/bash
#SBATCH --job-name=cats_mlp_20_21_22_23_24
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/cats_mlp_20_21_22_23_24.out
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=1
#SBATCH --time=05:00:00

project_path="$HOME/projects/JointChemicalModel"
experiment_script_path="$project_path/experiments/4.3_cats_mlp.py"

log_path="$project_path/results/logs"

source $HOME/anaconda3/etc/profile.d/conda.sh
export PYTHONPATH="$PYTHONPATH:$project_path"

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/cats_mlp/CHEMBL231_Ki -experiment 20 > "$log_path/cats_mlp_20.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/cats_mlp/CHEMBL264_Ki -experiment 21 > "$log_path/cats_mlp_21.log" &
pid2=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/cats_mlp/ESR1_ant -experiment 22 > "$log_path/cats_mlp_22.log" &
pid3=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/cats_mlp/CHEMBL1871_Ki -experiment 23 > "$log_path/cats_mlp_23.log" &
pid4=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/cats_mlp/TP53 -experiment 24 > "$log_path/cats_mlp_24.log" &
pid5=$!

wait $pid1
wait $pid2
wait $pid3
wait $pid4
wait $pid5

cp -r $project_path/results/cats_mlp/CHEMBL231_Ki /projects/prjs1021/JointChemicalModel/results/cats_mlp/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/cats_mlp/CHEMBL231_Ki
fi

cp -r $project_path/results/cats_mlp/CHEMBL264_Ki /projects/prjs1021/JointChemicalModel/results/cats_mlp/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/cats_mlp/CHEMBL264_Ki
fi

cp -r $project_path/results/cats_mlp/ESR1_ant /projects/prjs1021/JointChemicalModel/results/cats_mlp/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/cats_mlp/ESR1_ant
fi

cp -r $project_path/results/cats_mlp/CHEMBL1871_Ki /projects/prjs1021/JointChemicalModel/results/cats_mlp/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/cats_mlp/CHEMBL1871_Ki
fi

cp -r $project_path/results/cats_mlp/TP53 /projects/prjs1021/JointChemicalModel/results/cats_mlp/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/cats_mlp/TP53
fi

