#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 15:36:40 2024

@author: Qishuo

FAST_NN_ATE: data generate (linear case)

"""

# import packages
import numpy as np
import pandas as pd
import os


# set relative project path for the project 'Double_Deep_Learning'
path_project = os.path.dirname(__file__)


# set seed
seed = 2024
np.random.seed(seed)
# torch.manual_seed(seed)


# data generating functions
# generate factor
def generate_factor(p, r, n):
    B = np.random.uniform(low = -np.sqrt(3), high = np.sqrt(3), size = (p, r)) # loading matrix
    f = np.random.uniform(low = -1.0, high= 1.0, size = (n, r)) # latent factors
    u = np.random.uniform(low = -1.0, high = 1.0, size = (n, p)) # covariates
    return B, f, u


# generate propensity
def generate_propensity(f, u):
    pi = 1 / (1 + np.exp(-( f[:, 0] + f[:, 1] + f[:, 2] + f[:, 3] + u[:, 0] + u[:, 1] + u[:, 2] + u[:, 3] + u[:, 4] )))  # treatment assignment with size (n, )
    return pi


# generate covaraite effect
def generate_covariate_effect(f, u):
    mu = 10 + f[:, 0] + f[:, 1] + f[:, 2] + u[:, 0] + u[:, 1] + u[:, 2] + u[:, 3]
    return mu


# generate treatment effect
def generate_treatment_effect(f, u):
    tau = 5 + f[:, 3] + u[:, 4]
    return tau


# generate entire dataset
def generate_dataset(p, r, n, sigma_y):

    # generate factors
    B, f, u = generate_factor(p, r, n)

    # generate covariates
    X = f @ B.T + u # covariates with size (n, p)

    # generate treatment assignement
    pi = generate_propensity(f, u) # propensity model generation - treatment assignment with size (n, )
    T = np.zeros(n)
    for i in range(n):
        T[i] = np.random.binomial(1, pi[i], 1) # treatment assignment with size (n, )

    # generate outcome observed
    e_y = np.random.normal(loc = 0.0, scale = sigma_y, size = (n, )) # outcome model noise
    mu = generate_covariate_effect(f, u) # outcome model generation
    tau = generate_treatment_effect(f, u) # effect model generation
    Y = mu + tau * T + e_y # outcome observed with size (n, )

    return X, T, Y, pi, mu, tau, B, f, u


def concat_dataset(X, T, Y):
    dataset = pd.DataFrame(np.concatenate((X, T.reshape(-1, 1), Y.reshape(-1, 1)), axis=1))
    dataset.columns = ['X' + str(i) for i in range(X.shape[1])] + ['T', 'Y']
    return dataset


# only to check the preciseness of the codes
def concat_variable(pi, mu, tau):
    variable = pd.DataFrame(np.concatenate((pi.reshape(-1, 1), mu.reshape(-1, 1), tau.reshape(-1, 1)), axis=1))
    variable.columns = ['pi', 'mu', 'tau']
    return variable


def write_csv_dataset(p, r, n, sigma_y, simulation):
    for t in range(simulation):
        X, T, Y, pi, mu, tau, B, f, u = generate_dataset(p, r, n, sigma_y)
        dataset = concat_dataset(X, T, Y)
        path_outer = path_project + '/data_simulation/'
        path_inner = 'data_p_' + str(p) + '_sim_' + str(t) + '.csv'
        dataset.to_csv(path_outer + path_inner, index=False)
        print('p:' + str(p) + '; ' + 'simulation:' + str(t))
        # only to check the preciseness of the codes
        variable = concat_variable(pi, mu, tau)
        path_inner_variable = 'variable_p_' + str(p) + '_sim_' + str(t) + '.csv'
        variable.to_csv(path_outer + path_inner_variable, index=False)


# intialize parameter values - for low dimesional case
p_vec = [10, 50, 100, 500, 1000, 5000, 10000] # number of covariates
r = 4 # number of factors
sigma_y = 0.25 # outcome noice level
n = 1000 # number of observations
simulation = 5 # 100 # time of simulations

# data generating process
for p in p_vec:
    write_csv_dataset(p, r, n, sigma_y, simulation)
