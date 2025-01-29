#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 12:10:18 2024

@author: Qishuo

Double Deep Learning: script to run ATE estimation by Double Machine Learning (DML) estimator for application

"""

# import packages
import numpy as np
import pandas as pd
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utility.utility_data import read_dataset_csv_to_numpy
from utility.utility_data import data_split_X_T_Y
from utility.utility_data import variable_split_pi_mu_tau
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import os


# set relative project path for the project 'Double_Deep_Learning'
path_file = os.path.dirname(__file__)
path_file_parent = os.path.dirname(os.getcwd())

# run script on gpu if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set seed
seed = 2024
np.random.seed(seed)
torch.manual_seed(seed)


# intialize parameter value 
n = 5000 # syntehtic dataset size
simulation = 100 # time of simulations


# initialize parameter value - training model
epochs = 100
batchsize = 64
learning_rate = 0.0001
# r = 4
# r_bar = 10
L = 4
N = 300


# data and file path
path_data_outer = path_file + '/data_simulation/'
path_result_outer = path_file + '/result/'
path_variable_outer = path_file + '/variable/'


# run simulation
ATE_hat_mat = np.zeros(simulation)
ATE_ci_low_mat = np.zeros(simulation)
ATE_ci_up_mat = np.zeros(simulation)
ATE_true_mat = np.zeros(simulation)


coverage_count = 0

for t in range(simulation): 
    
    # import data
    print("simulation:" + str(t))
    path_outer_sim = path_file + '/data_simulation/'
    path_inner_sim = 'data_sim_' + str(t) + '.csv'
    dataset_sim = read_dataset_csv_to_numpy(path_outer_sim, path_inner_sim)
    path_inner_variable_sim = 'variable_sim_' + str(t) + '.csv'
    variable_sim = read_dataset_csv_to_numpy(path_outer_sim, path_inner_variable_sim)
    X, T, Y = data_split_X_T_Y(dataset_sim)
    pi, mu, tau = variable_split_pi_mu_tau(variable_sim)
    ATE_true = np.mean(tau)
    ATE_true_mat[t] = ATE_true
    
    # run functions
    estimator = CausalForestDML(model_y=RandomForestRegressor(n_estimators=100, max_depth=2),
                                model_t=RandomForestClassifier(n_estimators=100, max_depth=2),
                                discrete_treatment=True,
                                random_state=seed)
    estimator.fit(Y=Y, T=T, X=X, W=None)
    ATE_hat = estimator.ate(X)
    ATE_ci_low, ATE_ci_up = estimator.ate_interval(X, alpha=0.05)
    
    # save results & intermediate variables
    ATE_hat_mat[t] = ATE_hat
    ATE_ci_low_mat[t] =  ATE_ci_low
    ATE_ci_up_mat[t] =  ATE_ci_up
    if (ATE_ci_low <= ATE_true) and (ATE_true <= ATE_ci_up): 
        coverage_count += 1
    
MSE = sum(np.square(ATE_hat_mat - ATE_true_mat)) / simulation
coverage = coverage_count / simulation
final_results = np.zeros((1, 2))
final_results[0, 0] = MSE
final_results[0, 1] = coverage

  
# save results
path_inner_ATE = 'DML_ATE_hat.csv'
path_inner_ATE_ci_low = 'DML_ATE_ci_low.csv'
path_inner_ATE_ci_up = 'DML_ATE_ci_up.csv'
path_inner_final_results = 'DML_final_results.csv'
pd.DataFrame(ATE_hat_mat).to_csv(path_result_outer + path_inner_ATE, index=False)  
pd.DataFrame(ATE_ci_low_mat).to_csv(path_result_outer + path_inner_ATE_ci_low, index=False) 
pd.DataFrame(ATE_ci_up_mat).to_csv(path_result_outer + path_inner_ATE_ci_up, index=False)
pd.DataFrame(final_results).to_csv(path_result_outer + path_inner_final_results, index=False)

