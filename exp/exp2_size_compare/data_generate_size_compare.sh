#!/bin/bash
#SBATCH --job-name=data_generate_size_compare    # create a short name for your job
#SBATCH --output=logs/slurm-%A.%a.out            # stdout file
#SBATCH --error=logs/slurm-%A.%a.err             # stderr file
#SBATCH --nodes=1                                # node count
#SBATCH --ntasks=1                               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1                        # cpu-cores per task (>1 if multithread tasks)
#SBATCH --mem-per-cpu=16G                        # memory per cpu-core (4G is default)
#SBATCH --time=24:00:00                          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin                        # send email when process begins
#SBATCH --mail-type=fail                         # send email if job fails
#SBATCH --mail-type=end                          # send email when job ends
#SBATCH --mail-user=qy1448@princeton.edu

module purge
module load anaconda3/2024.2
conda activate Double_Deep_Learning_env

python data_generate_size_compare.py
