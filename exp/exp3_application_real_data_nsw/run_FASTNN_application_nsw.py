#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 12:55:35 2025

@author: Qishuo

Double Deep Learning: run FASTNN on real dataset application - NSW job datasets 

"""

# import packages
import numpy as np
import pandas as pd
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utility.utility_data import data_nsw_split_X_T_Y
from utility.utility_data import read_dataset_nsw_cleaned_to_numpy
from estimator.ddl_estimator import DDL
import os


# set relative project path for the project 'Double_Deep_Learning'
path_file = os.path.dirname(__file__)

# load datasets
# load experimental dataset
exp_name_vec = ['nsw_dw', 'psid1', 'psid2', 'psid3', 'cps', 'cps2', 'cps3']
path_outer = path_file + '/data_application/'
path_inner_control_vec = ['nswre74_control.txt', 'psid_controls.txt', 'psid2_controls.txt', 'psid3_controls.txt', 'cps_controls.txt', 'cps2_controls.txt', 'cps3_controls.txt']
path_save_outer = path_file + '/result/'
path_save_inner = 'application_nsw_FASTNN_result.csv'

ATE_hat_mat = np.zeros(len(exp_name_vec))
ATE_ci_low_mat = np.zeros(len(exp_name_vec))
ATE_ci_up_mat = np.zeros(len(exp_name_vec))


for k in range(len(exp_name_vec)): 
    exp_name = exp_name_vec[k]
    path_inner = 'data_application_' + exp_name + '.csv'
    data = read_dataset_nsw_cleaned_to_numpy(path_outer, path_inner)
    X, T, Y = data_nsw_split_X_T_Y(data)
    
    # run functions
    # initialize parameter value in training model for FASTNN
    estimator = DDL(X, T, Y, 
                    epochs = 200,
                    batchsize = 64,
                    learning_rate = 0.0001,
                    r = 1,
                    r_bar = 3,
                    L = 4,
                    N = 100)
    ATE_hat = estimator.ate_hat()
    ATE_ci_low, ATE_ci_up = estimator.ate_ci(tail='both', alpha=0.05)
    
    # save results & intermediate variables
    ATE_hat_mat[k] = ATE_hat
    ATE_ci_low_mat[k] =  ATE_ci_low
    ATE_ci_up_mat[k] =  ATE_ci_up
    
    print('experiment name: ' + exp_name)

#result = pd.DataFrame(np.concatenate((exp_name_vec, ATE_hat_mat, ATE_ci_low_mat, ATE_ci_up_mat)), 
#                                         columns=['exp name', 'ATE hat', 'CI low', 'CI up'])
#pd.DataFrame(result).to_csv(path_save_outer + path_save_inner, index=False)


