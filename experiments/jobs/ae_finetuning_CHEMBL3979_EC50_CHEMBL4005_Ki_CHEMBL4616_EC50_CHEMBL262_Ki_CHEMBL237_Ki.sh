#!/bin/bash
#SBATCH --job-name=ae_finetuning_CHEMBL3979_EC50_CHEMBL4005_Ki_CHEMBL4616_EC50_CHEMBL262_Ki_CHEMBL237_Ki
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/ae_finetuning_CHEMBL3979_EC50_CHEMBL4005_Ki_CHEMBL4616_EC50_CHEMBL262_Ki_CHEMBL237_Ki.out
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=1
#SBATCH --time=120:00:00

project_path="$HOME/projects/JointChemicalModel"
experiment_script_path="$project_path/experiments/4.5_ae_finetuning.py"

log_path="$project_path/results/logs"

source $HOME/anaconda3/etc/profile.d/conda.sh
export PYTHONPATH="$PYTHONPATH:$project_path"

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ae_finetuning/CHEMBL3979_EC50 -dataset CHEMBL3979_EC50 > "$log_path/ae_finetuning_CHEMBL3979_EC50.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ae_finetuning/CHEMBL4005_Ki -dataset CHEMBL4005_Ki > "$log_path/ae_finetuning_CHEMBL4005_Ki.log" &
pid2=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ae_finetuning/CHEMBL4616_EC50 -dataset CHEMBL4616_EC50 > "$log_path/ae_finetuning_CHEMBL4616_EC50.log" &
pid3=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ae_finetuning/CHEMBL262_Ki -dataset CHEMBL262_Ki > "$log_path/ae_finetuning_CHEMBL262_Ki.log" &
pid4=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ae_finetuning/CHEMBL237_Ki -dataset CHEMBL237_Ki > "$log_path/ae_finetuning_CHEMBL237_Ki.log" &
pid5=$!

wait $pid1
wait $pid2
wait $pid3
wait $pid4
wait $pid5

cp -r $project_path/results/ae_finetuning/CHEMBL3979_EC50 /projects/prjs1021/JointChemicalModel/results/ae_finetuning/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ae_finetuning/CHEMBL3979_EC50
fi

cp -r $project_path/results/ae_finetuning/CHEMBL4005_Ki /projects/prjs1021/JointChemicalModel/results/ae_finetuning/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ae_finetuning/CHEMBL4005_Ki
fi

cp -r $project_path/results/ae_finetuning/CHEMBL4616_EC50 /projects/prjs1021/JointChemicalModel/results/ae_finetuning/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ae_finetuning/CHEMBL4616_EC50
fi

cp -r $project_path/results/ae_finetuning/CHEMBL262_Ki /projects/prjs1021/JointChemicalModel/results/ae_finetuning/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ae_finetuning/CHEMBL262_Ki
fi

cp -r $project_path/results/ae_finetuning/CHEMBL237_Ki /projects/prjs1021/JointChemicalModel/results/ae_finetuning/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ae_finetuning/CHEMBL237_Ki
fi

