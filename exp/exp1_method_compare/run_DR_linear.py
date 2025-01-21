#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 13:04:07 2024

@author: Qishuo
"""

# import packages
import numpy as np
import pandas as pd
import torch

from utility.utility_data import read_dataset_to_numpy
from utility.utility_data import data_split_X_T_Y
from econml.dr import DRLearner
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# run script on gpu if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set seed
seed = 2024
np.random.seed(seed)
torch.manual_seed(seed)


# intialize parameter value
p_vec = [10, 50, 100, 500, 1000, 5000, 10000] # number of covariates
simulation = 5 # 100 # time of simulations
ATE_true = 5.0

# data and file path
data_path_outer = '/Users/Qishuo/Desktop/FAST_NN_ATE/scripts/data_simulation/data_simulation_linear/'
file_path_outer = '/Users/Qishuo/Desktop/FAST_NN_ATE/scripts/result/result_linear/'
#optional_path_outer = '/Users/Qishuo/Desktop/FAST_NN_ATE/scripts/variable/variable_linear/'


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
    for t in range(simulation): 
        print("t = " + str(t))
        print('-----------------------------------')
        print('-----------------------------------')
        
        # import data
        data = read_dataset_to_numpy(p, t, data_path_outer)
        X, T, Y = data_split_X_T_Y(data)
        
        # run functions
        estimator = DRLearner(model_regression=RandomForestRegressor(n_estimators=100, max_depth=50),
                              model_propensity=RandomForestClassifier(n_estimators=100, max_depth=50),
                              model_final=RandomForestRegressor(n_estimators=100, max_depth=2),
                              random_state=seed)
        estimator.fit(Y=Y, T=T, X=X, W=None)
        ATE_hat = estimator.ate(X)
        ITE_hat = estimator.effect(X)
        ATE_ci_low = np.percentile(ITE_hat, 2.5)
        ATE_ci_up = np.percentile(ITE_hat, 97.5)
        
        # save results
        ATE_hat_mat[k, t] = ATE_hat
        ATE_ci_low_mat[k, t] =  ATE_ci_low
        ATE_ci_up_mat[k, t] =  ATE_ci_up
        if (ATE_ci_low <= ATE_true) and (ATE_true <= ATE_ci_up): 
            coverage_count += 1
        
    MSE = sum(np.square(ATE_hat_mat[k, :] - ATE_true)) / simulation
    MSE_list[k] = MSE
    ATE_ci_low_mean = np.mean(ATE_ci_low_mat[k, :])
    ATE_ci_low_mean_list[k] = ATE_ci_low_mean
    ATE_ci_up_mean = np.mean(ATE_ci_up_mat[k, :])
    ATE_ci_up_mean_list[k] = ATE_ci_up_mean
    coverage = coverage_count / simulation
    coverage_list[k] = coverage
    
path_inner_ATE = 'DR_ATE_hat.csv'
path_inner_ATE_ci_low = 'DR_ATE_ci_low.csv'
path_inner_ATE_ci_up = 'DR_ATE_ci_up.csv'
path_inner_MSE = 'DR_MSE.csv'
path_inner_ATE_ci_low_mean = 'DR_ATE_ci_low_mean.csv'
path_inner_ATE_ci_up_mean = 'DR_ATE_ci_up_mean.csv'
path_inner_coverage = 'DR_coverage.csv'
pd.DataFrame(ATE_hat_mat).to_csv(file_path_outer + path_inner_ATE, index=False)  
pd.DataFrame(ATE_ci_low_mat).to_csv(file_path_outer + path_inner_ATE_ci_low, index=False) 
pd.DataFrame(ATE_ci_up_mat).to_csv(file_path_outer + path_inner_ATE_ci_up, index=False)
pd.DataFrame(MSE_list).to_csv(file_path_outer + path_inner_MSE, index=False)
pd.DataFrame(ATE_ci_low_mean_list).to_csv(file_path_outer + path_inner_ATE_ci_low_mean, index=False)
pd.DataFrame(ATE_ci_up_mean_list).to_csv(file_path_outer + path_inner_ATE_ci_up_mean, index=False)
pd.DataFrame(coverage_list).to_csv(file_path_outer + path_inner_coverage, index=False)

