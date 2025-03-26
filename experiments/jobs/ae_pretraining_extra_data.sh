#!/bin/bash
#SBATCH --job-name=ae_pretraining_extra_data
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/ae_pretraining_extra_data.out
#SBATCH -p gpu_a100
#SBATCH -N 1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=1
#SBATCH --time=80:00:00

project_path="$HOME/projects/JointChemicalModel"
experiment_script_path="$project_path/experiments/7.4_ae_pretraining_extra_data.py"

log_path="$project_path/results/logs"

source $HOME/anaconda3/etc/profile.d/conda.sh
export PYTHONPATH="$PYTHONPATH:$project_path"
$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/ae_pretraining_extra_data > "$log_path/ae_pretraining_extra_data.log"

cp -r $project_path/results/ae_pretraining_extra_data /projects/prjs1021/JointChemicalModel/results/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/ae_pretraining_extra_data
fi

