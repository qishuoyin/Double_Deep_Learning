#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 14:44:50 2025

@author: Qishuo

Double Deep Learning: creat multiple semi-synthetic datasets by shuffling

"""

# import packages
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utility.utility_data import read_dataset_csv_to_numpy
import os


# set relative project path for the project 'Double_Deep_Learning'
path_file = os.path.dirname(__file__)


# load cleaned real dataset for synthetics generation
path_outer = path_file + '/data_raw/'
path_inner = 'data_entire.csv'
path_inner_variable = 'variable_entire.csv'
dataset_entire = read_dataset_csv_to_numpy(path_outer, path_inner)
variable_entire = read_dataset_csv_to_numpy(path_outer, path_inner_variable)


# set simulation parameters
n_real = dataset_entire.shape[0]
simulation = 100
r = 4 # number of factors
sigma_y = 0.25 # outcome noice level


for t in range(simulation): 
    index = np.random.choice(range(n_real), size=1000, replace=False)
    dataset_sim = pd.DataFrame(dataset_entire[index, :])
    variable_sim = pd.DataFrame(variable_entire[index, :])
    
    path_outer_sim = path_file + '/data_simulation/'
    path_inner_sim = '/data_sim_' + str(t) + '.csv'
    dataset_sim.to_csv(path_outer_sim + path_inner_sim, index=False)
    path_inner_variable_sim = '/variable_sim_' + str(t) + '.csv'
    variable_sim.to_csv(path_outer_sim + path_inner_variable_sim, index=False)
    print("simulation:" + str(t))