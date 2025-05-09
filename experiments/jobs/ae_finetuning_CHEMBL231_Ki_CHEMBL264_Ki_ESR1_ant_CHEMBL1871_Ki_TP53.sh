#!/bin/bash
#SBATCH --job-name=ae_finetuning_CHEMBL231_Ki_CHEMBL264_Ki_ESR1_ant_CHEMBL1871_Ki_TP53
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/ae_finetuning_CHEMBL231_Ki_CHEMBL264_Ki_ESR1_ant_CHEMBL1871_Ki_TP53.out
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

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ae_finetuning/CHEMBL231_Ki -dataset CHEMBL231_Ki > "$log_path/ae_finetuning_CHEMBL231_Ki.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ae_finetuning/CHEMBL264_Ki -dataset CHEMBL264_Ki > "$log_path/ae_finetuning_CHEMBL264_Ki.log" &
pid2=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ae_finetuning/ESR1_ant -dataset ESR1_ant > "$log_path/ae_finetuning_ESR1_ant.log" &
pid3=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ae_finetuning/CHEMBL1871_Ki -dataset CHEMBL1871_Ki > "$log_path/ae_finetuning_CHEMBL1871_Ki.log" &
pid4=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ae_finetuning/TP53 -dataset TP53 > "$log_path/ae_finetuning_TP53.log" &
pid5=$!

wait $pid1
wait $pid2
wait $pid3
wait $pid4
wait $pid5

cp -r $project_path/results/ae_finetuning/CHEMBL231_Ki /projects/prjs1021/JointChemicalModel/results/ae_finetuning/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ae_finetuning/CHEMBL231_Ki
fi

cp -r $project_path/results/ae_finetuning/CHEMBL264_Ki /projects/prjs1021/JointChemicalModel/results/ae_finetuning/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ae_finetuning/CHEMBL264_Ki
fi

cp -r $project_path/results/ae_finetuning/ESR1_ant /projects/prjs1021/JointChemicalModel/results/ae_finetuning/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ae_finetuning/ESR1_ant
fi

cp -r $project_path/results/ae_finetuning/CHEMBL1871_Ki /projects/prjs1021/JointChemicalModel/results/ae_finetuning/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ae_finetuning/CHEMBL1871_Ki
fi

cp -r $project_path/results/ae_finetuning/TP53 /projects/prjs1021/JointChemicalModel/results/ae_finetuning/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ae_finetuning/TP53
fi

