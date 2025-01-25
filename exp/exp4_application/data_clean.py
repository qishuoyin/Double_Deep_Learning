#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 15:16:19 2025

@author: Qishuo

Double Deep Learning: data cleaning for real dataset application

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


# load original dataset
path_outer = path_file + '/data_original/'
path_inner = '2024-12.csv'
path_save_outer = path_file 
path_save_inner = '/data_real.csv'
data_original = read_dataset_csv_to_numpy(path_outer, path_inner)


# clean original dataset
data_drop_header = data_original[1:, 1:] # drop the header
data_float = data_drop_header.astype(np.float64) # convert dataset to float
data_drop_nan = data_float[:, ~np.isnan(data_float).any(axis=0)] # find columns with NaN values
data = np.transpose(data_drop_nan)
pd.DataFrame(data).to_csv(path_save_outer + path_save_inner, index=False)

