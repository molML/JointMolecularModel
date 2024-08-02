#!/bin/bash
#SBATCH --job-name=ecfp_mlp_0_1_2_3_4
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/ecfp_mlp_0_1_2_3_4.out
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

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ecfp_mlp/PPARG -experiment 0 > "$log_path/ecfp_mlp_0.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ecfp_mlp/CHEMBL244_Ki -experiment 1 > "$log_path/ecfp_mlp_1.log" &
pid2=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ecfp_mlp/CHEMBL204_Ki -experiment 2 > "$log_path/ecfp_mlp_2.log" &
pid3=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ecfp_mlp/CHEMBL218_EC50 -experiment 3 > "$log_path/ecfp_mlp_3.log" &
pid4=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ecfp_mlp/CHEMBL233_Ki -experiment 4 > "$log_path/ecfp_mlp_4.log" &
pid5=$!

wait $pid1
wait $pid2
wait $pid3
wait $pid4
wait $pid5

cp $project_path/results/ecfp_mlp/PPARG /projects/prjs1021/JointChemicalModel/results/ecfp_mlp/PPARG
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ecfp_mlp/PPARG
fi

cp $project_path/results/ecfp_mlp/CHEMBL244_Ki /projects/prjs1021/JointChemicalModel/results/ecfp_mlp/CHEMBL244_Ki
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ecfp_mlp/CHEMBL244_Ki
fi

cp $project_path/results/ecfp_mlp/CHEMBL204_Ki /projects/prjs1021/JointChemicalModel/results/ecfp_mlp/CHEMBL204_Ki
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ecfp_mlp/CHEMBL204_Ki
fi

cp $project_path/results/ecfp_mlp/CHEMBL218_EC50 /projects/prjs1021/JointChemicalModel/results/ecfp_mlp/CHEMBL218_EC50
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ecfp_mlp/CHEMBL218_EC50
fi

cp $project_path/results/ecfp_mlp/CHEMBL233_Ki /projects/prjs1021/JointChemicalModel/results/ecfp_mlp/CHEMBL233_Ki
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ecfp_mlp/CHEMBL233_Ki
fi

