#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 15:07:47 2024

@author: Qishuo

Double Deep Learning: script to run ATE estimation by GANITE

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
from ganite import Ganite
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
        model = Ganite(X, T, Y, num_iterations=500) # num_iterations=500 as default
        pred = model(X).cpu().numpy()
        ITE_hat = T*(Y-pred) + (1-T)*(pred-Y)
        ATE_hat = np.mean( ITE_hat )
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
    
path_inner_ATE = 'Ganite_ATE_hat.csv'
path_inner_ATE_ci_low = 'Ganite_ATE_ci_low.csv'
path_inner_ATE_ci_up = 'Ganite_ATE_ci_up.csv'
path_inner_MSE = 'Ganite_MSE.csv'
path_inner_ATE_ci_low_mean = 'Ganite_ATE_ci_low_mean.csv'
path_inner_ATE_ci_up_mean = 'Ganite_ATE_ci_up_mean.csv'
path_inner_coverage = 'Ganite_coverage.csv'
pd.DataFrame(ATE_hat_mat).to_csv(path_result_outer + path_inner_ATE, index=False)  
pd.DataFrame(ATE_ci_low_mat).to_csv(path_result_outer + path_inner_ATE_ci_low, index=False) 
pd.DataFrame(ATE_ci_up_mat).to_csv(path_result_outer + path_inner_ATE_ci_up, index=False)
pd.DataFrame(MSE_list).to_csv(path_result_outer + path_inner_MSE, index=False)
pd.DataFrame(ATE_ci_low_mean_list).to_csv(path_result_outer + path_inner_ATE_ci_low_mean, index=False)
pd.DataFrame(ATE_ci_up_mean_list).to_csv(path_result_outer + path_inner_ATE_ci_up_mean, index=False)
pd.DataFrame(coverage_list).to_csv(path_result_outer + path_inner_coverage, index=False)



