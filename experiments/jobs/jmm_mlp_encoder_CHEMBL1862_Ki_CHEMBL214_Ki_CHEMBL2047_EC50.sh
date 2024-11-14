#!/bin/bash
#SBATCH --job-name=jmm_mlp_encoder_CHEMBL1862_Ki_CHEMBL214_Ki_CHEMBL2047_EC50
#SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/jmm_mlp_encoder_CHEMBL1862_Ki_CHEMBL214_Ki_CHEMBL2047_EC50.out
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=1
#SBATCH --time=120:00:00

project_path="$HOME/projects/JointChemicalModel"
experiment_script_path="$project_path/experiments/4.7_jmm_mlp_encoder.py"

log_path="$project_path/results/logs"

source $HOME/anaconda3/etc/profile.d/conda.sh
export PYTHONPATH="$PYTHONPATH:$project_path"

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/jmm_mlp_encoder/CHEMBL1862_Ki -dataset CHEMBL1862_Ki > "$log_path/jmm_mlp_encoder_CHEMBL1862_Ki.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/jmm_mlp_encoder/CHEMBL214_Ki -dataset CHEMBL214_Ki > "$log_path/jmm_mlp_encoder_CHEMBL214_Ki.log" &
pid2=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o results/jmm_mlp_encoder/CHEMBL2047_EC50 -dataset CHEMBL2047_EC50 > "$log_path/jmm_mlp_encoder_CHEMBL2047_EC50.log" &
pid3=$!

wait $pid1
wait $pid2
wait $pid3

cp -r $project_path/results/jmm_mlp_encoder/CHEMBL1862_Ki /projects/prjs1021/JointChemicalModel/results/jmm_mlp_encoder/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/jmm_mlp_encoder/CHEMBL1862_Ki
fi

cp -r $project_path/results/jmm_mlp_encoder/CHEMBL214_Ki /projects/prjs1021/JointChemicalModel/results/jmm_mlp_encoder/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/jmm_mlp_encoder/CHEMBL214_Ki
fi

cp -r $project_path/results/jmm_mlp_encoder/CHEMBL2047_EC50 /projects/prjs1021/JointChemicalModel/results/jmm_mlp_encoder/
if [ $? -eq 0 ]; then
    rm -rf $project_path/results/jmm_mlp_encoder/CHEMBL2047_EC50
fi

