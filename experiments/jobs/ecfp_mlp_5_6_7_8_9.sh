#!/bin/bash
#SBATCH --job-name=ecfp_mlp_5_6_7_8_9
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/ecfp_mlp_5_6_7_8_9.out
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=1
#SBATCH --time=05:00:00

project_path="$HOME/projects/JointChemicalModel"
experiment_script_path="$project_path/experiments/4.2_ecfp_mlp.py"

log_path="$project_path/results/logs"

source $HOME/anaconda3/etc/profile.d/conda.sh
export PYTHONPATH="$PYTHONPATH:$project_path"

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ecfp_mlp/CHEMBL234_Ki -experiment 5 > "$log_path/ecfp_mlp_5.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ecfp_mlp/CHEMBL2971_Ki -experiment 6 > "$log_path/ecfp_mlp_6.log" &
pid2=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ecfp_mlp/CHEMBL2147_Ki -experiment 7 > "$log_path/ecfp_mlp_7.log" &
pid3=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ecfp_mlp/CHEMBL287_Ki -experiment 8 > "$log_path/ecfp_mlp_8.log" &
pid4=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ecfp_mlp/CHEMBL238_Ki -experiment 9 > "$log_path/ecfp_mlp_9.log" &
pid5=$!

wait $pid1
wait $pid2
wait $pid3
wait $pid4
wait $pid5

cp -r $project_path/results/ecfp_mlp/CHEMBL234_Ki /projects/prjs1021/JointChemicalModel/results/ecfp_mlp/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ecfp_mlp/CHEMBL234_Ki
fi

cp -r $project_path/results/ecfp_mlp/CHEMBL2971_Ki /projects/prjs1021/JointChemicalModel/results/ecfp_mlp/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ecfp_mlp/CHEMBL2971_Ki
fi

cp -r $project_path/results/ecfp_mlp/CHEMBL2147_Ki /projects/prjs1021/JointChemicalModel/results/ecfp_mlp/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ecfp_mlp/CHEMBL2147_Ki
fi

cp -r $project_path/results/ecfp_mlp/CHEMBL287_Ki /projects/prjs1021/JointChemicalModel/results/ecfp_mlp/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ecfp_mlp/CHEMBL287_Ki
fi

cp -r $project_path/results/ecfp_mlp/CHEMBL238_Ki /projects/prjs1021/JointChemicalModel/results/ecfp_mlp/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ecfp_mlp/CHEMBL238_Ki
fi

