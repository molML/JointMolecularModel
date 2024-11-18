#!/bin/bash
#SBATCH --job-name=ae_finetuning_CHEMBL4792_Ki_CHEMBL4203_Ki_CHEMBL219_Ki_CHEMBL228_Ki_CHEMBL2034_Ki
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/ae_finetuning_CHEMBL4792_Ki_CHEMBL4203_Ki_CHEMBL219_Ki_CHEMBL228_Ki_CHEMBL2034_Ki.out
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=1
#SBATCH --time=120:00:00

project_path="$HOME/projects/JointChemicalModel"
experiment_script_path="$project_path/experiments/4.7_ae_finetuning.py"

log_path="$project_path/results/logs"

source $HOME/anaconda3/etc/profile.d/conda.sh
export PYTHONPATH="$PYTHONPATH:$project_path"

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ae_finetuning/CHEMBL4792_Ki -dataset CHEMBL4792_Ki > "$log_path/ae_finetuning_CHEMBL4792_Ki.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ae_finetuning/CHEMBL4203_Ki -dataset CHEMBL4203_Ki > "$log_path/ae_finetuning_CHEMBL4203_Ki.log" &
pid2=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ae_finetuning/CHEMBL219_Ki -dataset CHEMBL219_Ki > "$log_path/ae_finetuning_CHEMBL219_Ki.log" &
pid3=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ae_finetuning/CHEMBL228_Ki -dataset CHEMBL228_Ki > "$log_path/ae_finetuning_CHEMBL228_Ki.log" &
pid4=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ae_finetuning/CHEMBL2034_Ki -dataset CHEMBL2034_Ki > "$log_path/ae_finetuning_CHEMBL2034_Ki.log" &
pid5=$!

wait $pid1
wait $pid2
wait $pid3
wait $pid4
wait $pid5

cp -r $project_path/results/ae_finetuning/CHEMBL4792_Ki /projects/prjs1021/JointChemicalModel/results/ae_finetuning/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ae_finetuning/CHEMBL4792_Ki
fi

cp -r $project_path/results/ae_finetuning/CHEMBL4203_Ki /projects/prjs1021/JointChemicalModel/results/ae_finetuning/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ae_finetuning/CHEMBL4203_Ki
fi

cp -r $project_path/results/ae_finetuning/CHEMBL219_Ki /projects/prjs1021/JointChemicalModel/results/ae_finetuning/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ae_finetuning/CHEMBL219_Ki
fi

cp -r $project_path/results/ae_finetuning/CHEMBL228_Ki /projects/prjs1021/JointChemicalModel/results/ae_finetuning/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ae_finetuning/CHEMBL228_Ki
fi

cp -r $project_path/results/ae_finetuning/CHEMBL2034_Ki /projects/prjs1021/JointChemicalModel/results/ae_finetuning/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ae_finetuning/CHEMBL2034_Ki
fi

