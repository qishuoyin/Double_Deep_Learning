#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 16:54:42 2024

@author: Qishuo

FAST_NN_ATE: read dataset and split variables

"""

# import packages
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigs


# utility function
# read csv to numpy array
def read_dataset_to_numpy(p, t, path_outer): 
    #path_outer = '/Users/Qishuo/Desktop/FAST_NN_ATE/scripts/data_simulation_linear/'
    path_inner = 'data_p_' + str(p) + '_sim_' + str(t) + '.csv'
    data_df = pd.read_csv(path_outer + path_inner)
    data_np = data_df.to_numpy()
    print('[read data] p:' + str(p) + '; ' + 'simulation:' + str(t))
    return data_np


# split dataset into covariate(x), treatment(t), and outcome(y) by column
def data_split_X_T_Y(data):
    X = data[:, :-2]
    T = data[:, -2]
    Y = data[:, -1]
    return X, T, Y


# compute matrix parameters in FAST model
def FASTmat(X, p, r, r_bar): 
    cov_mat = np.matmul(np.transpose(X), X)
    eigen_values, eigen_vectors = eigs(cov_mat, r_bar, which='LM')
    dp_mat = eigen_vectors / np.sqrt(p)

    eigen_values_oracler, eigen_vectors_oracler = eigs(cov_mat, r, which='LM')
    dp_matrix_oracler = eigen_vectors_oracler / np.sqrt(p)
    estimate_f = np.matmul(X, dp_mat)
    cov_f_mat = np.matmul(np.transpose(estimate_f), estimate_f)
    cov_fx_mat = np.matmul(np.transpose(estimate_f), X)
    rs_matrix = np.matmul(np.linalg.pinv(cov_f_mat), cov_fx_mat)
    
    return dp_mat, rs_matrix


# only to check the preciseness of the codes
def read_variable_to_numpy(p, t, path_outer): 
    #path_outer = '/Users/Qishuo/Desktop/FAST_NN_ATE/scripts/data_simulation_linear/'
    path_inner_variable = 'variable_p_' + str(p) + '_sim_' + str(t) + '.csv'
    variable_df = pd.read_csv(path_outer + path_inner_variable)
    variable_np = variable_df.to_numpy()
    return variable_np


def variable_split_pi_mu_tau(variable):
    pi = variable[:, 0]
    mu = variable[:, 1]
    tau = variable[:, 2]
    return pi, mu, tau
