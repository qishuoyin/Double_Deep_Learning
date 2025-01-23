#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 18:47:46 2025

@author: Qishuo

Double Deep Learning: data cleaning for real dataset application - NSW job datasets

"""

# import packages
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utility.utility_data import read_dataset_nsw_to_numpy
from utility.utility_data import concat_treat_control
import os


# set relative project path for the project 'Double_Deep_Learning'
path_file = os.path.dirname(__file__)

# load datasets
# load experimental dataset
exp_name_vec = ['nsw_dw', 'psid1', 'psid2', 'psid3', 'cps', 'cps2', 'cps3']
path_outer = path_file + '/data_original/'
path_inner_treat = 'nswre74_treated.txt'
path_inner_control_vec = ['nswre74_control.txt', 'psid_controls.txt', 'psid2_controls.txt', 'psid3_controls.txt', 'cps_controls.txt', 'cps2_controls.txt', 'cps3_controls.txt']
path_save_outer = path_file + '/data_application/'

for exp_name, path_inner_control in zip(exp_name_vec, path_inner_control_vec): 
    data_control = read_dataset_nsw_to_numpy(path_outer, path_inner_control)
    data_treat = read_dataset_nsw_to_numpy(path_outer, path_inner_treat)
    data = concat_treat_control(data_treat, data_control)
    path_save_inner = 'data_application_' + exp_name + '.csv'
    pd.DataFrame(data).to_csv(path_save_outer + path_save_inner, index=False)
    






