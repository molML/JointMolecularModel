#!/bin/bash
#SBATCH --job-name=ecfp_mlp_PPARG_CHEMBL244_Ki_CHEMBL204_Ki_CHEMBL218_EC50_CHEMBL233_Ki
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/ecfp_mlp_PPARG_CHEMBL244_Ki_CHEMBL204_Ki_CHEMBL218_EC50_CHEMBL233_Ki.out
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

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ecfp_mlp/PPARG -dataset PPARG > "$log_path/ecfp_mlp_PPARG.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ecfp_mlp/CHEMBL244_Ki -dataset CHEMBL244_Ki > "$log_path/ecfp_mlp_CHEMBL244_Ki.log" &
pid2=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ecfp_mlp/CHEMBL204_Ki -dataset CHEMBL204_Ki > "$log_path/ecfp_mlp_CHEMBL204_Ki.log" &
pid3=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ecfp_mlp/CHEMBL218_EC50 -dataset CHEMBL218_EC50 > "$log_path/ecfp_mlp_CHEMBL218_EC50.log" &
pid4=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ecfp_mlp/CHEMBL233_Ki -dataset CHEMBL233_Ki > "$log_path/ecfp_mlp_CHEMBL233_Ki.log" &
pid5=$!

wait $pid1
wait $pid2
wait $pid3
wait $pid4
wait $pid5

cp -r $project_path/results/ecfp_mlp/PPARG /projects/prjs1021/JointChemicalModel/results/ecfp_mlp/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ecfp_mlp/PPARG
fi

cp -r $project_path/results/ecfp_mlp/CHEMBL244_Ki /projects/prjs1021/JointChemicalModel/results/ecfp_mlp/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ecfp_mlp/CHEMBL244_Ki
fi

cp -r $project_path/results/ecfp_mlp/CHEMBL204_Ki /projects/prjs1021/JointChemicalModel/results/ecfp_mlp/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ecfp_mlp/CHEMBL204_Ki
fi

cp -r $project_path/results/ecfp_mlp/CHEMBL218_EC50 /projects/prjs1021/JointChemicalModel/results/ecfp_mlp/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ecfp_mlp/CHEMBL218_EC50
fi

cp -r $project_path/results/ecfp_mlp/CHEMBL233_Ki /projects/prjs1021/JointChemicalModel/results/ecfp_mlp/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ecfp_mlp/CHEMBL233_Ki
fi

