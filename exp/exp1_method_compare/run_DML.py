#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 12:10:18 2024

@author: Qishuo

Double Deep Learning: script to run ATE estimation by Double Machine Learning (DML) estimator

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
p_vec = [10, 50, 100, 500, 1000, 5000, 10000] # number of covariates
simulation = 100 # time of simulations
ATE_true = 5.0

# data and file path
path_data_outer = path_file + '/data_simulation/'
path_result_outer = path_file + '/result/'


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
        data = read_dataset_to_numpy(p, t, path_data_outer)
        X, T, Y = data_split_X_T_Y(data)
        
        # run functions
        estimator = CausalForestDML(model_y=RandomForestRegressor(n_estimators=100, max_depth=2),
                                    model_t=RandomForestClassifier(n_estimators=100, max_depth=2),
                                    discrete_treatment=True,
                                    random_state=seed)
        estimator.fit(Y=Y, T=T, X=X, W=None)
        ATE_hat = estimator.ate(X)
        ATE_ci_low, ATE_ci_up = estimator.ate_interval(X, alpha=0.05)
        
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
    
path_inner_ATE = 'DML_ATE_hat.csv'
path_inner_ATE_ci_low = 'DML_ATE_ci_low.csv'
path_inner_ATE_ci_up = 'DML_ATE_ci_up.csv'
path_inner_MSE = 'DML_MSE.csv'
path_inner_ATE_ci_low_mean = 'DML_ATE_ci_low_mean.csv'
path_inner_ATE_ci_up_mean = 'DML_ATE_ci_up_mean.csv'
path_inner_coverage = 'DML_coverage.csv'
pd.DataFrame(ATE_hat_mat).to_csv(path_result_outer + path_inner_ATE, index=False)  
pd.DataFrame(ATE_ci_low_mat).to_csv(path_result_outer + path_inner_ATE_ci_low, index=False) 
pd.DataFrame(ATE_ci_up_mat).to_csv(path_result_outer + path_inner_ATE_ci_up, index=False)
pd.DataFrame(MSE_list).to_csv(path_result_outer + path_inner_MSE, index=False)
pd.DataFrame(ATE_ci_low_mean_list).to_csv(path_result_outer + path_inner_ATE_ci_low_mean, index=False)
pd.DataFrame(ATE_ci_up_mean_list).to_csv(path_result_outer + path_inner_ATE_ci_up_mean, index=False)
pd.DataFrame(coverage_list).to_csv(path_result_outer + path_inner_coverage, index=False)



