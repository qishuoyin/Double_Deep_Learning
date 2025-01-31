# Double Deep Learning Package and Experiment Implementation
This is the repository for the package and experiment implementation for the project (paper) Double Deep Learning in Average Treatment Effect (ATE) estimation. This README file gives a brief introduction to the setup of the environment, the replication of the experiments, and the usage of the package. 

## Table of Contents

- [Introduction](#introduction)
- [Environment](#Envrionment)
- [Example](#Example)

## Introduction
This repository contains four subfolders: 

(1) ```model```: it contains two files for the model structures inside the Double Deep Learning estimator
- ```model_FASTNN.py``` : gives the architecture of the Factor Augmented Sparse Throughput Deep ReLU Neural Networks (FAST-NN) inside the Double Deep Learning estimator.
- ```model_VanillaNN.py``` : gives the architecture of the Vanilla Neural Networks inside the Double Deep Learning estimator.

(2) ```estimator```: it contains four files to implement the Double Deep Learning method in steps: 
- ```dataloader.py``` : builds the data loader to load input data to pytorch tensors.
- ```train_model.py``` : contains functions for neural network model fitting and prediction.
- ```dr_estimator.py``` : gives the structure of the doubly robust estimator.
- ```ddl_estimator.py``` : gives the structure of the double deep learning estimator. 

(3) ```exp```: it contains three subfolders for the experiments in the project (paper). both ```.py``` and ```.sh``` files are given to implement methods and submit jobs to clusters. 
- ```exp1_method_compare``` : contains files to generate datasets and implement experiments comparing Double Deep Learning (DDL), Vanilla Neural Networks with L2 regularization (Vanilla-NN), Generative Adversarial Nets for inference of Individualized Treatment Effects (GANITE), Causal Forest (CF), Double Robust Forest Model (DR), and Double Machine Learning Forest Model (DML) on fixed sample size datasets with various covariate dimension. 
- ```exp2_size_compare``` : contains files to generate datasets and implement experiments comparing the performance of Double Deep Learning (DDL) on datasets with various sample sizes. 
- ```exp3_application_image``` : contains files to implement experiments comparing Double Deep Learning (DDL), Vanilla Neural Networks with L2 regularization (Vanilla-NN), Generative Adversarial Nets for inference of Individualized Treatment Effects (GANITE), Causal Forest (CF), Double Robust Forest Model (DR), and Double Machine Learning Forest Model (DML) on a semi-synthetic dataset by [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. 

(4) ```utility```: it contains ```utility_data.py``` to read and prepare datasets for the experiments and functions to compute the hyper-parameters in the implemented methods. 


## Environment
The information to set up the environment *Double Deep Learning* of this repository can be found in the file ```Double_Deep_Learning.yml```, which can be directly installed by 
```
conda env create -f environment.yml
```
because the first line of the ```.yml``` file sets the new environment's name. 
For simplicity, one can also set the environment by
```
pip install numpy==1.26.4
pip install pandas==2.2.3
pip install torch==2.5.1
pip install matplotlib==3.10.0
pip install scipy==1.15.1
pip install econml==0.15.1
pip install sklearn==1.5.2
pip install ganite
```

Meanwhile, if people want to conduct their own experiments with the use of this repository, one of the easiest ways is to create a subfolder under the folder ```exp``` and be sure to include the following commands to call the utility functions and the double deep learning estimator
```
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
```
People can also run experiments anywhere else, but they may want to change the relative paths set in this repository a little bit by editing the command 
```
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
```
Printing out the current directories is always helpful by executing the following chunk. 
```
path_file = os.path.dirname(__file__)
path_file_parent = os.path.dirname(os.getcwd())
print("current directory: " + path_file)
print("parent directory:" + path_file_parent)
```

## Example
We give a simple example of implementing the double deep learning algorithm by leveraging our functions here. Suppose the new python file is created under a  subfolder of ```exp``` and the covariate $X$, treatment $T$, and outcome $Y$ are already loaded as NumPy arrays. Then, one can estimate the average treatment effect of this dataset by executing the following chunk
```
import numpy as np
import pandas as pd
import torch
from estimator.ddl_estimator import DDL

# estimate ATE and its confidence interval (CI)
estimator = DDL(X, T, Y)
ATE_hat, ATE_ci_low, ATE_ci_up = estimator.ate_hat_ci(tail='both', alpha=0.05)
```
You may change the confidence interval to be one-sided by changing the value of ```tail``` to be ```'left'``` or ```'right'```. Meanwhile, the confidence interval can also be changed by altering the value of the variable ```alpha```. 
