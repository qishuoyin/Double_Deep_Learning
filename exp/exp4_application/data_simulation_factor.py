#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 18:47:19 2025

@author: Qishuo

Double Deep Learning: generate semi-synthetic datasets by real dataset

"""

# import packages
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utility.utility_data import read_dataset_csv_to_numpy
from sklearn.decomposition import PCA
import os


# set relative project path for the project 'Double_Deep_Learning'
path_file = os.path.dirname(__file__)


# load cleaned real dataset for synthetics generation
path_outer = path_file + '/data_raw/'
path_inner = 'data_cleaned.csv'
path_save_inner = 'data_entire.csv'
path_save_inner_variable = 'variable_entire.csv'
data_real = read_dataset_csv_to_numpy(path_outer, path_inner)
data_real_transpose = np.transpose(data_real)
X = data_real_transpose - np.mean(data_real_transpose, axis=0)
# X = X[:, 0:1000] # first try a fraction of total dimensions locally

# set simulation parameters
n_real = data_real_transpose.shape[0]
r = 4 # number of factors
sigma_y = 0.25 # outcome noice level


# data synthetic functions
# generate factors, loading matrix and idiosyncratic component
def generate_factor(X, r): 
    # Sigma = np.cov(X)
    # eigenvalues, eigenvectors = np.linalg.eig(Sigma)
    Sigma = X @ np.transpose(X)
    pca = PCA()
    pca.fit(Sigma)
    # print("fit PCA")
    eigenvalues = pca.explained_variance_
    # print("eigenvalue shape0:" + str(eigenvalues.shape[0]) )
    # print("got eigen value")
    eigenvectors = pca.components_
    # print("eigenvector shape0:" + str(eigenvectors.shape[0]) + "shape1:" + str(eigenvectors.shape[1]) )
    # print("got eigen vector")
    f = np.sqrt(n_real) * eigenvectors[:, 0:r] # latent factors
    # print("f shape0:" + str(f.shape[0]) + "shape1" + str(f.shape[1]) )
    B = 1/n_real * np.transpose(X) @ f # loading matrix
    u = X - f @ np.transpose(B) # idiosyncratic component
    return B, f, u


# generate propensity
def generate_propensity(f, u): 
    pi = 1 / (1 + np.exp(-( np.sin(f[:, 0]) + f[:, 1] + f[:, 2] + f[:, 3] + np.sin(u[:, 0]) + u[:, 1] + u[:, 2] + u[:, 3] + u[:, 4] )))
    return pi


# generate covaraite effect
def generate_covariate_effect(f, u):
    mu = 10 + f[:, 0] + np.sin(f[:, 1]) + f[:, 2]*f[:, 3] + u[:, 0] * (u[:, 1] + np.sin(u[:, 2])) + u[:, 3] + u[:, 4]
    return mu


# generate treatment effect
def generate_treatment_effect(f, u):
    tau = f[:, 0]*(f[:, 1] + 3) + f[:, 2] + np.sin(f[:, 3]) + np.sin(u[:, 0]) + u[:, 1] + u[:, 2]*u[:, 3]*u[:, 4]
    return tau


# generate entire dataset
def generate_dataset(X, r):

    # generate factors
    B, f, u = generate_factor(X, r)
    
    # generate treatment assignement
    pi = generate_propensity(f, u) # propensity model generation - treatment assignment with size (n_real, )
    T = np.zeros(n_real)
    for i in range(n_real):
        T[i] = np.random.binomial(1, pi[i], 1) # treatment assignment with size (n_real, )

    # generate outcome observed
    e_y = np.random.normal(loc = 0.0, scale = sigma_y, size = (n_real, )) # outcome model noise
    mu = generate_covariate_effect(f, u) # outcome model generation
    tau = generate_treatment_effect(f, u) # effect model generation
    Y = mu + tau * T + e_y # outcome observed with size (n_real, )
    return X, T, Y, pi, mu, tau, B, f, u


def concat_dataset(X, T, Y):
    # dataset = np.concatenate((X, T.reshape(-1, 1), Y.reshape(-1, 1)), axis=1)
    dataset = pd.DataFrame(np.concatenate((X, T.reshape(-1, 1), Y.reshape(-1, 1)), axis=1))
    dataset.columns = ['X' + str(i) for i in range(X.shape[1])] + ['T', 'Y']
    return dataset


# only to check the preciseness of the codes
def concat_variable(pi, mu, tau):
    # variable = np.concatenate((pi.reshape(-1, 1), mu.reshape(-1, 1), tau.reshape(-1, 1)), axis=1)
    variable = pd.DataFrame(np.concatenate((pi.reshape(-1, 1), mu.reshape(-1, 1), tau.reshape(-1, 1)), axis=1))
    variable.columns = ['pi', 'mu', 'tau']
    return variable


def write_csv_dataset_entire(X, r):
    X, T, Y, pi, mu, tau, B, f, u = generate_dataset(X, r)
    dataset = concat_dataset(X, T, Y)
    # np.savetxt(path_outer + path_save_inner, dataset, delimiter=",")
    dataset.to_csv(path_outer + path_save_inner, index=False)
    variable = concat_variable(pi, mu, tau)
    # np.savetxt(path_outer + path_save_inner_variable, variable, delimiter=",")
    variable.to_csv(path_outer + path_save_inner_variable, index=False)
    return dataset, variable
    

dataset_entire, variable_entire = write_csv_dataset_entire(X, r)
# print("entire dataset loaded")

    
    
    
    