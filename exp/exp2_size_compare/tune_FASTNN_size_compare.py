#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 15:50:47 2025

@author: Qishuo

Double Deep Learning: tune NN parameters - width and depth - for FASTNN model based on different sample size and covariate dimension

"""

# import packages
import numpy as np
import pandas as pd
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utility.utility_data import read_dataset_size_compare_to_numpy
from utility.utility_data import data_split_X_T_Y
from estimator.ddl_estimator import DDL
import os
import itertools


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
p_vec = [10, 100, 200, 500, 1000, 2000, 5000] # number of covariates
n_vec = [1000, 2000, 3000, 4000, 5000] # size of dataset
# n_vec = [100, 500, 1000, 5000] # size of dataset


sim_tune = 5 # time of simulations
ATE_true = 5.0
# initialize parameter value - training model
epochs = 200
batchsize = 64
learning_rate = 0.0001
r = 4
r_bar = 10
# L = 4
# N = 300


# TRY1: intialize parameter for grid search
param_grid = {
    'L': [2, 3, 4, 5], 
    'N': [100, 200, 300, 400, 500]
    }


# TRY2: intialize parameter for grid search
# param_grid = {
#     'L': [3, 5, 6, 8], 
#     'N': [300, 400, 500, 600, 800]
#     }


# data and file path
path_data_outer = path_file + '/data_simulation/'
path_result_outer = path_file + '/result/'
path_variable_outer = path_file + '/variable/'


# simulation
MSE_list = np.zeros(len(p_vec))
ATE_ci_low_mean_list = np.zeros(len(p_vec))
ATE_ci_up_mean_list = np.zeros(len(p_vec))
coverage_list = np.zeros(len(p_vec))


with open('tune_FASTNN_size_compare_printed_results.txt', 'w') as file: 
    
    for l in range(len(n_vec)): 
        n = n_vec[l]
        print("n = " + str(n))
        print('-----------------------------------')
        print('-----------------------------------')
        
        # simulation for given sample size
        ATE_hat_mat = np.zeros((len(p_vec), sim_tune))
        ATE_ci_low_mat = np.zeros((len(p_vec), sim_tune))
        ATE_ci_up_mat = np.zeros((len(p_vec), sim_tune))
        
        for k in range(len(p_vec)): 
            p = p_vec[k]
            print("p = " + str(p))
            print('-----------------------------------')
            print('-----------------------------------')
            
            coverage_count = 0
            
            # only to check the preciseness of the codes
            pi_hat_mat = np.zeros((n, sim_tune))
            mu_hat_mat = np.zeros((n, sim_tune))
            tau_hat_mat = np.zeros((n, sim_tune))
            
            for t in range(sim_tune): 
                print("t = " + str(t))
                print('-----------------------------------')
                print('-----------------------------------')
                
                best_mse = 100000
                best_params = None
                
                for params in itertools.product(*param_grid.values()):
                    params_dict = dict(zip(param_grid.keys(), params))
                
                    # import data
                    data = read_dataset_size_compare_to_numpy(p, n, t, path_data_outer)
                    X, T, Y = data_split_X_T_Y(data)
                    
                    # run functions
                    estimator = DDL(X, T, Y, L=params_dict['L'], N=params_dict['N'])
                    ATE_hat, ATE_ci_low, ATE_ci_up = estimator.ate_hat_ci(tail='both', alpha=0.05)
                    pi_hat = estimator.pi_hat()
                    mu_hat = estimator.mu_hat()
                    tau_hat = estimator.tau_hat()
                    Y_hat = estimator.Y_hat()
                    
                    mse = sum(np.square(T - pi_hat.reshape(-1))) / len(T) + sum(np.square(Y - Y_hat)) / len(T)
                    
                    if mse < best_mse:
                        best_mse = mse
                        best_params = params_dict
                
                line1 = f"sample size, n = {n}; covariate dimension, p = {p}; sim time, t = {t};"
                file.write(line1 + "\n")
                line2 = f"best parameters: {best_params}; best mse = {best_mse}"
                file.write(line2 + "\n")
                    
                    # save results & intermediate variables
                    # ATE_hat_mat[k, t] = ATE_hat
                    # ATE_ci_low_mat[k, t] =  ATE_ci_low
                    # ATE_ci_up_mat[k, t] =  ATE_ci_up
                    # if (ATE_ci_low <= ATE_true) and (ATE_true <= ATE_ci_up): 
                    #     coverage_count += 1
                    
                    # only to check the preciseness of the codes
                    # pi_hat_mat[:, t:(t+1)] = pi_hat
                    # mu_hat_mat[:, t:(t+1)] = mu_hat
                    # tau_hat_mat[:, t:(t+1)] = tau_hat
                    
                # path_inner_pi_hat = 'FAST_pi_hat_p_' + str(p) + '_n_' + str(n) + '.csv'
                # path_inner_mu_hat = 'FAST_mu_hat_p_' + str(p) + '_n_' + str(n) + '.csv'
                # path_inner_tau_hat = 'FAST_tau_hat_p_' + str(p) + '_n_' + str(n) + '.csv'
                
                # pd.DataFrame(pi_hat_mat).to_csv(path_variable_outer + path_inner_pi_hat, index=False)  # optinal intermediate result 
                # pd.DataFrame(mu_hat_mat).to_csv(path_variable_outer + path_inner_mu_hat, index=False)  # optinal intermediate result 
                # pd.DataFrame(tau_hat_mat).to_csv(path_variable_outer + path_inner_tau_hat, index=False)  # optinal intermediate result 
                
                # MSE = sum(np.square(ATE_hat_mat[k, :] - ATE_true)) / sim_tune
                # MSE_list[k] = MSE
                # ATE_ci_low_mean = np.mean(ATE_ci_low_mat[k, :])
                # ATE_ci_low_mean_list[k] = ATE_ci_low_mean
                # ATE_ci_up_mean = np.mean(ATE_ci_up_mat[k, :])
                # ATE_ci_up_mean_list[k] = ATE_ci_up_mean
                # coverage = coverage_count / sim_tune
                # coverage_list[k] = coverage
                
            # path_inner_ATE = 'FAST_ATE_hat_n_' + str(n) + '.csv'
            # path_inner_ATE_ci_low = 'FAST_ATE_ci_low_n_' + str(n) + '.csv'
            # path_inner_ATE_ci_up = 'FAST_ATE_ci_up_n_' + str(n) + '.csv'
            # path_inner_MSE = 'FAST_MSE_n_' + str(n) + '.csv'
            # path_inner_ATE_ci_low_mean = 'FAST_ATE_ci_low_mean_n_' + str(n) + '.csv'
            # path_inner_ATE_ci_up_mean = 'FAST_ATE_ci_up_mean_n_' + str(n) + '.csv'
            # path_inner_coverage = 'FAST_coverage_n_' + str(n) + '.csv'
            # pd.DataFrame(ATE_hat_mat).to_csv(path_result_outer + path_inner_ATE, index=False)  
            # pd.DataFrame(ATE_ci_low_mat).to_csv(path_result_outer + path_inner_ATE_ci_low, index=False) 
            # pd.DataFrame(ATE_ci_up_mat).to_csv(path_result_outer + path_inner_ATE_ci_up, index=False)
            # pd.DataFrame(MSE_list).to_csv(path_result_outer + path_inner_MSE, index=False)
            # pd.DataFrame(ATE_ci_low_mean_list).to_csv(path_result_outer + path_inner_ATE_ci_low_mean, index=False)
            # pd.DataFrame(ATE_ci_up_mean_list).to_csv(path_result_outer + path_inner_ATE_ci_up_mean, index=False)
            # pd.DataFrame(coverage_list).to_csv(path_result_outer + path_inner_coverage, index=False)



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

