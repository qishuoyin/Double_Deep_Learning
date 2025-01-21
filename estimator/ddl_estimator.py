#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 20:26:30 2024

@author: Qishuo

FAST_NN_ATE: double robust deep learning factor model ATE estimator

"""

# import packages
import numpy as np
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utility.utility_data import FASTmat
from model.model_FASTNN import FactorAugmentedSparseThroughputNN
from model.model_VanillaNN import VanillaNetBase
from estimator.train_model import NNModelTrain
from estimator.dr_estimator import DoubleRobustEst

# Double Robust Deep Learning Factor Model ATE estimator
class DDL: 
    def __init__(self, 
                 X, T, Y, 
                 epochs=100, 
                 batchsize=64, 
                 learning_rate=0.0001, 
                 L=4, 
                 N=300, 
                 factor=True, 
                 r=4, 
                 r_bar=10, 
                 sparsity=None, 
                 regularization_type=None, 
                 lambda_reg=1, 
                 penalty_weight=None, 
                 reg_tau=0.005): 
        
        self.X = X
        self.T = T
        self.Y = Y
        self.epochs = epochs
        self.batchsize = batchsize
        self.learning_rate = learning_rate
        self.L = L
        self.N = N
        self.factor = factor
        self.r = r
        self.r_bar = r_bar
        self.sparsity = sparsity
        self.regularization_type=regularization_type, 
        self.lambda_reg=lambda_reg, 
        self.penalty_weight=penalty_weight, 
        self.reg_tau=reg_tau
        
    def DDL_est(self): 
        n = self.X.shape[0] # sample size
        p = self.X.shape[1] # covariates dimension
        dp_mat, rs_mat = FASTmat(self.X, p, self.r, self.r_bar)
        
        if self.factor == True: # condut FAST NN
            model_propensity = FactorAugmentedSparseThroughputNN(input_dim=p, r_bar=self.r_bar, depth=self.L, width=self.N, dp_mat=dp_mat, sparsity=self.r_bar, rs_mat=rs_mat, logistic=True)
            loss_fn_propensity = model_propensity.least_square_loss()
            optimizer_propensity = torch.optim.Adam(model_propensity.parameters(), lr=self.learning_rate)
            estimator_propensity = NNModelTrain(self.epochs, self.batchsize, model_propensity, loss_fn_propensity, optimizer_propensity)
            
            model_outcome = FactorAugmentedSparseThroughputNN(input_dim=p, r_bar=self.r_bar, depth=self.L, width=self.N, dp_mat=dp_mat, sparsity=self.r_bar, rs_mat=rs_mat, logistic=False)
            loss_fn_outcome = model_outcome.least_square_loss()
            optimizer_outcome = torch.optim.Adam(model_outcome.parameters(), lr=self.learning_rate)
            estimator_outcome = NNModelTrain(self.epochs, self.batchsize, model_outcome, loss_fn_outcome, optimizer_outcome)
            est = DoubleRobustEst(self.X, self.T, self.Y, self.batchsize, estimator_propensity, estimator_outcome, estimator_outcome, penalty_weight=1.5*np.log(p)/n, reg_tau=0.005) 
            
        elif self.factor == False: # condut Vanilla NN
            model_propensity = VanillaNetBase(input_dim=p, depth=self.L, width=self.N, logistic=True)
            loss_fn_propensity = model_propensity.least_square_loss()
            optimizer_propensity = torch.optim.Adam(model_propensity.parameters(), lr=self.learning_rate)
            estimator_propensity = NNModelTrain(self.epochs, self.batchsize, model_propensity, loss_fn_propensity, optimizer_propensity)
            
            model_outcome = VanillaNetBase(input_dim=p, depth=self.L, width=self.N, logistic=False)
            loss_fn_outcome = model_outcome.least_square_loss()
            optimizer_outcome = torch.optim.Adam(model_outcome.parameters(), lr=self.learning_rate)
            estimator_outcome = NNModelTrain(self.epochs, self.batchsize, model_outcome, loss_fn_outcome, optimizer_outcome)
            est = DoubleRobustEst(self.X, self.T, self.Y, self.batchsize, estimator_propensity, estimator_outcome, estimator_outcome, regularization_type='L2') 
        
        return est
    
    
    def ate_hat(self): 
        est = self.DDL_est()
        return est.dr_ate_est()
    
    
    def ate_ci(self, tail='both', alpha=0.05): 
        est = self.DDL_est()
        return est.ate_ci_est(tail, alpha)
    
    
    def pi_hat(self): 
        est = self.DDL_est()
        return est.propensity_est()
    
    
    def mu_hat(self): 
        est = self.DDL_est()
        return est.mu_est()
    
    
    def tau_hat(self): 
        est = self.DDL_est()
        return est.tau_est()
    
        