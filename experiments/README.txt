To replicate the study, you could run all of these scripts to get from the raw original data to the results presented in the paper.

I do not redistribute the raw datasets (you can find them from their respective papers), but I do provide the processed data in a Zenodo repo.
With this data, you can skip the first step '0_clean_data.py'. Simply put the 'data' directory into the code repo root.

All scripts that run torch models rely on a SLURM-based computer cluster with Nvidia A100 gpus. I often fit multiple jobs
on one GPU since they have plenty of memory to save some compute credits. I provided the SLURM job scripts in
the 'jobs' directory, alongside the config files in the 'hyperparams' directory.

However, since this is just the code I ran myself to get the results for the paper, I am sure that the training scripts
will not work out of the box with your setup. You will have to adjust the SLURM scripts and paths to make things work.