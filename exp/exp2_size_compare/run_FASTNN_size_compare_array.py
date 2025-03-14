#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 17:26:06 2024

@author: Qishuo

Double Deep Learning: script to compare ATE estimation by FASTNN model with different sample size
Written in job array to run on a cluster

"""

import os
import numpy as np
import pandas as pd
import torch
import sys
from pathlib import Path
from itertools import product  # Allows generating (n, p) pairs

# Import custom utility functions
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utility.utility_data import read_dataset_size_compare_to_numpy, data_split_X_T_Y
from estimator.ddl_estimator import DDL

# Define hyperparameter grids
n_vec = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]  # Dataset sizes
p_vec = [10, 100, 200, 500, 1000, 2000, 5000]  # Number of covariates

# Generate a list of all (n, p) combinations
param_grid = list(product(n_vec, p_vec))  # [(n1, p1), (n1, p2), ..., (n10, p7)]

# Get the SLURM array index
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])  # SLURM assigns a unique ID

# Extract (n, p) pair based on task ID
n, p = param_grid[idx]

# Print task info
print(f"Running SLURM Task ID: {idx} with (n={n}, p={p})")

# Set random seed
seed = 2024
np.random.seed(seed)
torch.manual_seed(seed)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Experiment parameters
simulation = 100  # Number of simulations
ATE_true = 5.0
epochs = 100
batchsize = 64
learning_rate = 0.0001
r = 4
r_bar = 10
L = 4
N = 400

# Define paths for results
path_file = os.path.dirname(__file__)
path_data_outer = os.path.join(path_file, 'data_simulation')
path_result_outer = os.path.join(path_file, 'result')
path_variable_outer = os.path.join(path_file, 'variable')

# Ensure output directories exist
os.makedirs(path_variable_outer, exist_ok=True)
os.makedirs(path_result_outer, exist_ok=True)

# Storage arrays for simulation results
MSE_list = np.zeros(simulation)
ATE_ci_low_mean_list = np.zeros(simulation)
ATE_ci_up_mean_list = np.zeros(simulation)
coverage_list = np.zeros(simulation)

ATE_hat_mat = np.zeros(simulation)
ATE_ci_low_mat = np.zeros(simulation)
ATE_ci_up_mat = np.zeros(simulation)

coverage_count = 0

# only to check the preciseness of the codes
pi_hat_mat = np.zeros(simulation)
mu_hat_mat = np.zeros(simulation)
tau_hat_mat = np.zeros(simulation)

# Run simulations
for t in range(simulation):
    print(f"Simulation {t} for (n={n}, p={p})")

    # Load dataset
    data = read_dataset_size_compare_to_numpy(p, n, t, path_data_outer)
    X, T, Y = data_split_X_T_Y(data)

    # Initialize estimator and compute estimates
    estimator = DDL(X, T, Y)
    ATE_hat, ATE_ci_low, ATE_ci_up = estimator.ate_hat_ci(tail='both', alpha=0.05)

    # Store results
    ATE_hat_mat[t] = ATE_hat
    ATE_ci_low_mat[t] = ATE_ci_low
    ATE_ci_up_mat[t] = ATE_ci_up
    if ATE_ci_low <= ATE_true <= ATE_ci_up:
        coverage_count += 1

    # only to check the preciseness of the codes
    pi_hat_mat[t:(t+1)] = pi_hat
    mu_hat_mat[t:(t+1)] = mu_hat
    tau_hat_mat[t:(t+1)] = tau_hat

# Compute statistics
MSE = np.mean((ATE_hat_mat - ATE_true) ** 2)
coverage = coverage_count / simulation
ATE_ci_low_mean = np.mean(ATE_ci_low_mat)
ATE_ci_up_mean = np.mean(ATE_ci_up_mat)

# Save results
path_inner_ATE = 'FAST_ATE_hat_n_' + str(n) + '.csv'
path_inner_ATE_ci_low = 'FAST_ATE_ci_low_n_' + str(n) + '.csv'
path_inner_ATE_ci_up = 'FAST_ATE_ci_up_n_' + str(n) + '.csv'
path_inner_MSE = 'FAST_MSE_n_' + str(n) + '.csv'
path_inner_ATE_ci_low_mean = 'FAST_ATE_ci_low_mean_n_' + str(n) + '.csv'
path_inner_ATE_ci_up_mean = 'FAST_ATE_ci_up_mean_n_' + str(n) + '.csv'
path_inner_coverage = 'FAST_coverage_n_' + str(n) + '.csv'
pd.DataFrame(ATE_hat_mat).to_csv(path_result_outer + path_inner_ATE, index=False)  
pd.DataFrame(ATE_ci_low_mat).to_csv(path_result_outer + path_inner_ATE_ci_low, index=False) 
pd.DataFrame(ATE_ci_up_mat).to_csv(path_result_outer + path_inner_ATE_ci_up, index=False)
pd.DataFrame(MSE_list).to_csv(path_result_outer + path_inner_MSE, index=False)
pd.DataFrame(ATE_ci_low_mean_list).to_csv(path_result_outer + path_inner_ATE_ci_low_mean, index=False)
pd.DataFrame(ATE_ci_up_mean_list).to_csv(path_result_outer + path_inner_ATE_ci_up_mean, index=False)
pd.DataFrame(coverage_list).to_csv(path_result_outer + path_inner_coverage, index=False)

print(f"Finished task {idx} with (n={n}, p={p})")

