#!/bin/bash
#SBATCH --job-name=run_FASTNN_size_compare_array      # create a short name for your job
#SBATCH --output=logs/slurm-%A.%a.out                 # stdout file
#SBATCH --error=logs/slurm-%A.%a.err                  # stderr file
#SBATCH --nodes=1                                     # node count
#SBATCH --ntasks=1                                    # total number of tasks across all nodes
#SBATCH --cpus-per-task=8                             # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=64G                             # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1                                  # number of gpus per node
#SBATCH --time=72:00:00                               # total run time limit (HH:MM:SS)
#SBATCH --array=0-69                                  # job array with index values 0, 1, 2, ..., 9
#SBATCH --mail-type=all                               # send email on job start, end and fault
#SBATCH --mail-user=qy1448@princeton.edu

echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Executing on machine:" $(hostname)

module purge
module load anaconda3/2024.2
conda activate Double_Deep_Learning_env

python run_FASTNN_size_compare_array.py
