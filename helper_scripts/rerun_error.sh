#!/bin/bash
#SBATCH --job-name=hugo_job
#SBATCH --cpus-per-task=1
#SBATCH --output=./slurm_logs/rerun_error.out
#SBATCH --error=./slurm_logs/rerun_error.err
#SBATCH --array=1-1
#SBATCH --time=30:00
#SBATCH --mem-per-cpu=2G
#SBATCH --account=def-whitem
#SBATCH --dependency=singleton

. $HOME/my_cc_scripts/custom_commands.sh
CC_MY_SINGULARITY_EXEC pipenv run python -m helper_scripts.debug_find_error_runs > $HOME/my_cc_scripts/sbatch_files/${CLUSTER_NAME}.txt
CC_SBATCH_FROM_FILE --time 4:00:00 --mem_per_cpu 8G

