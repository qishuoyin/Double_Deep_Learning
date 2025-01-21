#!/bin/bash
#SBATCH --job-name=run_FASTNN_size_compare_array     # create a short name for your job
#SBATCH --output=slurm-%A.%a.out                      # stdout file
#SBATCH --error=slurm-%A.%a.err                       # stderr file
#SBATCH --nodes=1                                     # node count
#SBATCH --ntasks=1                                    # total number of tasks across all nodes
#SBATCH --cpus-per-task=1                             # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G                              # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1                                  # number of gpus per node
#SBATCH --time=24:00:00                               # total run time limit (HH:MM:SS)
#SBATCH --array=0-5                                   # job array with index values 0, 1, 2, 3, 4, 5
#SBATCH --mail-type=all                               # send email on job start, end and fault
#SBATCH --mail-user=qy1448@princeton.edu

echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $(hostname)

module purge
module load anaconda3/2024.2
conda activate Double_Deep_Learning_env

python run_FASTNN_size_compare_array.py 
