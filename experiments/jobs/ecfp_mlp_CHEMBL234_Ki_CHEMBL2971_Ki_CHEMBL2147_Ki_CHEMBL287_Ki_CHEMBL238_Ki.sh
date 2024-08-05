#!/bin/bash
#SBATCH --job-name=ecfp_mlp_CHEMBL234_Ki_CHEMBL2971_Ki_CHEMBL2147_Ki_CHEMBL287_Ki_CHEMBL238_Ki
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/ecfp_mlp_CHEMBL234_Ki_CHEMBL2971_Ki_CHEMBL2147_Ki_CHEMBL287_Ki_CHEMBL238_Ki.out
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=1
#SBATCH --time=32:00:00

project_path="$HOME/projects/JointChemicalModel"
experiment_script_path="$project_path/experiments/4.2_ecfp_mlp.py"

log_path="$project_path/results/logs"

source $HOME/anaconda3/etc/profile.d/conda.sh
export PYTHONPATH="$PYTHONPATH:$project_path"

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ecfp_mlp/CHEMBL234_Ki -dataset CHEMBL234_Ki > "$log_path/ecfp_mlp_CHEMBL234_Ki.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ecfp_mlp/CHEMBL2971_Ki -dataset CHEMBL2971_Ki > "$log_path/ecfp_mlp_CHEMBL2971_Ki.log" &
pid2=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ecfp_mlp/CHEMBL2147_Ki -dataset CHEMBL2147_Ki > "$log_path/ecfp_mlp_CHEMBL2147_Ki.log" &
pid3=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ecfp_mlp/CHEMBL287_Ki -dataset CHEMBL287_Ki > "$log_path/ecfp_mlp_CHEMBL287_Ki.log" &
pid4=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ecfp_mlp/CHEMBL238_Ki -dataset CHEMBL238_Ki > "$log_path/ecfp_mlp_CHEMBL238_Ki.log" &
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

