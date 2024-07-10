# !/bin/bash
# SBATCH --job-name=vae_pretraining_95_96_97_98_99
# SBATCH --output=/home/tilborgd/projects/JointChemicalModel/results/out/vae_pretraining_95_96_97_98_99.out
# SBATCH -p gpu
# SBATCH -N 1
# SBATCH --ntasks=18
# SBATCH --gpus-per-node=1
# SBATCH --time=120:00:00

experiment_name="vae_pretraining"

project_path="$HOME/projects/JointChemicalModel"
experiment_script_path="$project_path/experiments/3.0_vae_pretraining.py"

out_path="$project_path/results/$experiment_name"
log_path="$project_path/results/logs"

source $HOME/anaconda3/etc/profile.d/conda.sh
export PYTHONPATH="$PYTHONPATH:$project_path"

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o $out_path -experiment 95 > "$log_path/${experiment_name}_95.log" &
pid1=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o $out_path -experiment 96 > "$log_path/${experiment_name}_96.log" &
pid2=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o $out_path -experiment 97 > "$log_path/${experiment_name}_97.log" &
pid3=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o $out_path -experiment 98 > "$log_path/${experiment_name}_98.log" &
pid4=$!

$HOME/anaconda3/envs/karman/bin/python -u $experiment_script_path -o $out_path -experiment 99 > "$log_path/${experiment_name}_99.log" &
pid5=$!

wait $pid1
wait $pid2
wait $pid3
wait $pid4
wait $pid5

