#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 17:08:55 2024

@author: Qishuo

Double Deep Learning: script to run ATE estimation by Vanilla-L2 model

"""

# import packages
import numpy as np
import pandas as pd
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utility.utility_data import read_dataset_to_numpy
from utility.utility_data import data_split_X_T_Y
from estimator.ddl_estimator import DDL
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
p_vec = [10, 50, 100, 500, 1000, 5000, 10000] # number of covariates
simulation = 100 # time of simulations
ATE_true = 5.0
# initialize parameter value - training model
epochs = 100
batchsize = 64
learning_rate = 0.0001
r = 4
r_bar = 10
L = 4
N = 300


# data and file path
path_data_outer = path_file + '/data_simulation/'
path_result_outer = path_file + '/result/'
path_variable_outer = path_file + '/variable/'


# simulation for low dimensional case
ATE_hat_mat = np.zeros((len(p_vec), simulation))
ATE_ci_low_mat = np.zeros((len(p_vec), simulation))
ATE_ci_up_mat = np.zeros((len(p_vec), simulation))
MSE_list = np.zeros(len(p_vec))
ATE_ci_low_mean_list = np.zeros(len(p_vec))
ATE_ci_up_mean_list = np.zeros(len(p_vec))
coverage_list = np.zeros(len(p_vec))

for k in range(len(p_vec)): 
    p = p_vec[k]
    print("p = " + str(p))
    print('-----------------------------------')
    print('-----------------------------------')
    
    coverage_count = 0
    
    # only to check the preciseness of the codes
    n = 1000 # number of observations
    pi_hat_mat = np.zeros((n, simulation))
    mu_hat_mat = np.zeros((n, simulation))
    tau_hat_mat = np.zeros((n, simulation))
    
    for t in range(simulation): 
        print("t = " + str(t))
        print('-----------------------------------')
        print('-----------------------------------')
        
        # import data
        data = read_dataset_to_numpy(p, t, path_data_outer)
        X, T, Y = data_split_X_T_Y(data)
        
        # run functions
        estimator = DDL(X, T, Y, factor=False)
        ATE_hat, ATE_ci_low, ATE_ci_up = estimator.ate_hat_ci(tail='both', alpha=0.05)
        pi_hat = estimator.pi_hat()
        mu_hat = estimator.mu_hat()
        tau_hat = estimator.tau_hat()
        
        # save results & intermediate variables
        ATE_hat_mat[k, t] = ATE_hat
        ATE_ci_low_mat[k, t] =  ATE_ci_low
        ATE_ci_up_mat[k, t] =  ATE_ci_up
        if (ATE_ci_low <= ATE_true) and (ATE_true <= ATE_ci_up): 
            coverage_count += 1
        
        # only to check the preciseness of the codes
        pi_hat_mat[:, t:(t+1)] = pi_hat
        mu_hat_mat[:, t:(t+1)] = mu_hat
        tau_hat_mat[:, t:(t+1)] = tau_hat
        
    path_inner_pi_hat = 'Vanilla_pi_hat_p_' + str(p) + '.csv'
    path_inner_mu_hat = 'Vanilla_mu_hat_p_' + str(p) + '.csv'
    path_inner_tau_hat = 'Vanilla_tau_hat_p_' + str(p) + '.csv'
    
    pd.DataFrame(pi_hat_mat).to_csv(path_variable_outer + path_inner_pi_hat, index=False)  # optinal intermediate result 
    pd.DataFrame(mu_hat_mat).to_csv(path_variable_outer + path_inner_mu_hat, index=False)  # optinal intermediate result 
    pd.DataFrame(tau_hat_mat).to_csv(path_variable_outer + path_inner_tau_hat, index=False)  # optinal intermediate result 
    
    MSE = sum(np.square(ATE_hat_mat[k, :] - ATE_true)) / simulation
    MSE_list[k] = MSE
    ATE_ci_low_mean = np.mean(ATE_ci_low_mat[k, :])
    ATE_ci_low_mean_list[k] = ATE_ci_low_mean
    ATE_ci_up_mean = np.mean(ATE_ci_up_mat[k, :])
    ATE_ci_up_mean_list[k] = ATE_ci_up_mean
    coverage = coverage_count / simulation
    coverage_list[k] = coverage
    
path_inner_ATE = 'Vanilla_L2_ATE_hat.csv'
path_inner_ATE_ci_low = 'Vanilla_L2_ATE_ci_low.csv'
path_inner_ATE_ci_up = 'Vanilla_L2_ATE_ci_up.csv'
path_inner_MSE = 'Vanilla_L2_MSE.csv'
path_inner_ATE_ci_low_mean = 'Vanilla_L2_ATE_ci_low_mean.csv'
path_inner_ATE_ci_up_mean = 'Vanilla_L2_ATE_ci_up_mean.csv'
path_inner_coverage = 'Vanilla_L2_coverage.csv'
pd.DataFrame(ATE_hat_mat).to_csv(path_result_outer + path_inner_ATE, index=False)  
pd.DataFrame(ATE_ci_low_mat).to_csv(path_result_outer + path_inner_ATE_ci_low, index=False) 
pd.DataFrame(ATE_ci_up_mat).to_csv(path_result_outer + path_inner_ATE_ci_up, index=False)
pd.DataFrame(MSE_list).to_csv(path_result_outer + path_inner_MSE, index=False)
pd.DataFrame(ATE_ci_low_mean_list).to_csv(path_result_outer + path_inner_ATE_ci_low_mean, index=False)
pd.DataFrame(ATE_ci_up_mean_list).to_csv(path_result_outer + path_inner_ATE_ci_up_mean, index=False)
pd.DataFrame(coverage_list).to_csv(path_result_outer + path_inner_coverage, index=False)
