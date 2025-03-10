#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 02:42:15 2025

@author: Qishuo

Double Deep Learning: script for size compare visualization

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


# set relative project path for the project 'Double_Deep_Learning'
path_file = os.path.dirname(__file__)
path_file_parent = os.path.dirname(os.getcwd())

# Define file paths
file_paths = {
    'n=1000': path_file + '/result/FAST_MSE_n_1000.csv',
    'n=2000': path_file + '/result/FAST_MSE_n_2000.csv',
    'n=3000': path_file + '/result/FAST_MSE_n_3000.csv',
    'n=4000': path_file + '/result/FAST_MSE_n_4000.csv',
    'n=5000': path_file + '/result/FAST_MSE_n_5000.csv',
}

# Define the number of covariate dimensions corresponding to the MSE values
p_vec = [10, 50, 100, 500, 1000, 5000, 10000]
# n_vec = [1000, 2000, 3000, 4000, 5000] # size of dataset

# Load data and process MSE values
data = {}
for method, path in file_paths.items():
    df = pd.read_csv(path, header=None, names=["MSE"])
    df = df.iloc[1:8]  # Remove the first row and keep the next 7
    df["Dimension"] = p_vec  # Assign corrected dimensions as x-axis values
    data[method] = df.reset_index(drop=True)  # Reset index

# Merge data based on corrected dimensions
merged_data = pd.DataFrame({"Dimension": data["n=1000"]["Dimension"]})
for method, df in data.items():
    merged_data[method] = df["MSE"]

# Plot MSE curves with corrected x-axis values
plt.figure(figsize=(10, 6))
for method in file_paths.keys():
    plt.plot(p_vec, merged_data[method], marker='o', label=method, linewidth=2)

# Beautify the plot
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Covariate Dimension", fontsize=14)
plt.ylabel("Mean Squared Error (MSE)", fontsize=14)
# plt.title("Comparison of MSE Across Different Dimensions", fontsize=16)
plt.legend(title="Methods", fontsize=12)
# plt.grid(True, linestyle="--", alpha=0.6)
plt.xticks(p_vec, labels=p_vec, fontsize=12)  # Corrected x-axis values
plt.yticks(fontsize=12)


# Save the plot
plot_path = path_file + '/size_compare_mse_plot.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')


# Show the plot
plt.show()
