#!/bin/bash
#SBATCH --job-name=cats_mlp_CHEMBL4792_Ki_CHEMBL4203_Ki_CHEMBL219_Ki_CHEMBL228_Ki_CHEMBL2034_Ki
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/cats_mlp_CHEMBL4792_Ki_CHEMBL4203_Ki_CHEMBL219_Ki_CHEMBL228_Ki_CHEMBL2034_Ki.out
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

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/cats_mlp/CHEMBL4792_Ki -dataset CHEMBL4792_Ki > "$log_path/cats_mlp_CHEMBL4792_Ki.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/cats_mlp/CHEMBL4203_Ki -dataset CHEMBL4203_Ki > "$log_path/cats_mlp_CHEMBL4203_Ki.log" &
pid2=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/cats_mlp/CHEMBL219_Ki -dataset CHEMBL219_Ki > "$log_path/cats_mlp_CHEMBL219_Ki.log" &
pid3=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/cats_mlp/CHEMBL228_Ki -dataset CHEMBL228_Ki > "$log_path/cats_mlp_CHEMBL228_Ki.log" &
pid4=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/cats_mlp/CHEMBL2034_Ki -dataset CHEMBL2034_Ki > "$log_path/cats_mlp_CHEMBL2034_Ki.log" &
pid5=$!

wait $pid1
wait $pid2
wait $pid3
wait $pid4
wait $pid5

cp -r $project_path/results/cats_mlp/CHEMBL4792_Ki /projects/prjs1021/JointChemicalModel/results/cats_mlp/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/cats_mlp/CHEMBL4792_Ki
fi

cp -r $project_path/results/cats_mlp/CHEMBL4203_Ki /projects/prjs1021/JointChemicalModel/results/cats_mlp/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/cats_mlp/CHEMBL4203_Ki
fi

cp -r $project_path/results/cats_mlp/CHEMBL219_Ki /projects/prjs1021/JointChemicalModel/results/cats_mlp/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/cats_mlp/CHEMBL219_Ki
fi

cp -r $project_path/results/cats_mlp/CHEMBL228_Ki /projects/prjs1021/JointChemicalModel/results/cats_mlp/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/cats_mlp/CHEMBL228_Ki
fi

cp -r $project_path/results/cats_mlp/CHEMBL2034_Ki /projects/prjs1021/JointChemicalModel/results/cats_mlp/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/cats_mlp/CHEMBL2034_Ki
fi

