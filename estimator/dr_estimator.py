#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 16:39:40 2024

@author: Qishuo

Double Deep Learning: double robust ATE estimator

"""

import numpy as np
import scipy.stats
from sklearn.model_selection import train_test_split 
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from estimator.dataloader import data_to_torch, data_torch_dataloader


# Double Robust ATE estimator
class DoubleRobustEst:
    
    '''
    A class to coduct double robust average treatment effect (ATE) estimation
    
    ...
    Methods
    -------
    __init__()
        initialize the module
    propensity_est()
        estimate propensity score
    '''
    
    def __init__(self,
                 X, T, Y,
                 batchsize,
                 estimator_propensity,
                 estimator_outcome_treat,
                 estimator_outcome_control, 
                 regularization_type=None, 
                 lambda_reg=1, 
                 penalty_weight=None, 
                 reg_tau=0.005
                 # return_CI=False, 
                 # return_intermediate=False, 
                 # return_pi_hat=False, 
                 # return_mu_hat=False, 
                 # return_tau_hat=False
                 ):
        
        self.X = X
        self.T = T
        self.Y = Y
        self.batchsize = batchsize
        self.estimator_propensity = estimator_propensity
        self.estimator_outcome_treat = estimator_outcome_treat
        self.estimator_outcome_control = estimator_outcome_control
        self.regularization_type = regularization_type
        self.lambda_reg=lambda_reg
        self.penalty_weight = penalty_weight
        self.reg_tau = reg_tau
        # self.return_CI=return_CI
        # self.return_intermediate=return_intermediate
        # self.return_pi_hat=return_pi_hat
        # self.return_mu_hat=return_mu_hat
        # self.return_tau_hat=return_tau_hat
        
        
    def propensity_est(self): 
        
        '''
        A function to estimate propensity score

        Returns
        -------
        pi_hat : numpy.ndarray
            (n, ) matrix of propensity estimation
        '''
        
        X = self.X
        T = self.T
        X_train, X_val, T_train, T_val = train_test_split(X, T, test_size=0.2)
        # torch_propensity = data_to_torch(X, T.reshape(-1, 1))
        torch_propensity_train = data_to_torch(X_train, T_train.reshape(-1, 1))
        torch_propensity_val = data_to_torch(X_val, T_val.reshape(-1, 1))
        # loader_propensity = data_torch_dataloader(torch_propensity, self.batchsize)
        loader_propensity_train = data_torch_dataloader(torch_propensity_train, self.batchsize)
        loader_propensity_val = data_torch_dataloader(torch_propensity_val, self.batchsize)
        
        propensity_fitted = self.estimator_propensity.fit(loader_propensity_train, loader_propensity_val, logistic=True,  regularization_type=self.regularization_type, penalty_weight=self.penalty_weight) # fit propensity model
        pi_hat = self.estimator_propensity.predict(propensity_fitted, X).cpu().numpy() # predict propensity score
        
        # adjust propensity estimation by truncation
        pi_hat[pi_hat < 0.01] = 0.01
        pi_hat[pi_hat > 0.99] = 0.99
        return pi_hat
    
    
    def outcome_treat_est(self): 
        
        '''
        A function to estimate outcome for treatment

        Returns
        -------
        mu1_hat : numpy.ndarray
            (number of T==1, ) matrix of outcome for treatment estimation
        '''
        
        X = self.X
        X_treat = self.X[self.T == 1]
        Y_treat = self.Y[self.T == 1]
        X_treat_train, X_treat_val, Y_treat_train, Y_treat_val = train_test_split(X_treat, Y_treat, test_size=0.2)
        # torch_outcome_treat = data_to_torch(X_treat, Y_treat.reshape(-1, 1))
        torch_outcome_treat_train = data_to_torch(X_treat_train, Y_treat_train.reshape(-1, 1))
        torch_outcome_treat_val = data_to_torch(X_treat_val, Y_treat_val.reshape(-1, 1))
        # loader_outcome_treat = data_torch_dataloader(torch_outcome_treat, self.batchsize)
        loader_outcome_treat_train = data_torch_dataloader(torch_outcome_treat_train, self.batchsize)
        loader_outcome_treat_val = data_torch_dataloader(torch_outcome_treat_val, self.batchsize)
        
        outcome_treat_fitted = self.estimator_outcome_treat.fit(loader_outcome_treat_train, loader_outcome_treat_val, logistic=False, regularization_type=self.regularization_type, penalty_weight=self.penalty_weight)  # outcome for treatment model
        mu1_hat = self.estimator_outcome_treat.predict(outcome_treat_fitted, X).cpu().numpy() # predict outcome for treatment
        return mu1_hat
    
    
    def outcome_control_est(self): 
        
        '''
        A function to estimate outcome for control

        Returns
        -------
        mu0_hat : numpy.ndarray
            (number of T==0, ) matrix of outcome for control estimation
        '''
        
        X = self.X
        X_control = self.X[self.T == 0]
        Y_control = self.Y[self.T == 0]
        X_control_train, X_control_val, Y_control_train, Y_control_val = train_test_split(X_control, Y_control, test_size=0.2)
        # torch_outcome_control = data_to_torch(X_control, Y_control.reshape(-1, 1))
        torch_outcome_control_train = data_to_torch(X_control_train, Y_control_train.reshape(-1, 1))
        torch_outcome_control_val = data_to_torch(X_control_val, Y_control_val.reshape(-1, 1))
        # loader_outcome_control = data_torch_dataloader(torch_outcome_control, self.batchsize)
        loader_outcome_control_train = data_torch_dataloader(torch_outcome_control_train, self.batchsize)
        loader_outcome_control_val = data_torch_dataloader(torch_outcome_control_val, self.batchsize)
        
        outcome_control_fitted = self.estimator_outcome_control.fit(loader_outcome_control_train, loader_outcome_control_val, logistic=False, regularization_type=self.regularization_type, penalty_weight=self.penalty_weight) # outcome for control model
        mu0_hat = self.estimator_outcome_control.predict(outcome_control_fitted, X).cpu().numpy() # predict outcome for control
        return mu0_hat
    
    
    def mu_est(self): 
        
        '''
        A function to estimate effect from covariates

        Returns
        -------
        mu_hat : numpy.ndarray
            (n, ) matrix of effect from covariates estimation
        '''
        
        return self.outcome_control_est()
    
    
    def tau_est(self): 
        
        '''
        A function to estimate effect from treatment

        Returns
        -------
        tau_hat : numpy.ndarray
            (n, ) matrix of effect from treatment estimation
        '''
        
        return self.outcome_treat_est() - self.outcome_control_est()
    
    
    def dr_ite_est(self): 
        
        '''
        A function to estimate double robust individual treatment effect

        Returns
        -------
        ite_hat : numpy.ndarray
            (n, ) matrix of individual treatment effect estimation
        '''
        
        pi_hat = self.propensity_est()
        mu1_hat = self.outcome_treat_est()
        mu0_hat = self.outcome_control_est()
        ite_hat = (np.multiply(self.T, self.Y) / pi_hat.reshape(-1,) - np.multiply((1 - self.T), self.Y) / (1 - pi_hat.reshape(-1,))) - ( self.T - pi_hat.reshape(-1,) ) * ( mu1_hat.reshape(-1,) / pi_hat.reshape(-1,) + mu0_hat.reshape(-1,) / (1 - pi_hat.reshape(-1,)) )
        return ite_hat
    
    
    def ate_est(self, tail='both', alpha=0.05): 
        
        '''
        A function to compute confidence interval of average treatment effect (ATE) estimation
        
        Parameters
        ----------
        tail : str (default to be 'both')
            whether the confidence interval is both tailed or one tailed
            'no' for not return confidence interval only return ATE estimation
            'both' for two tailed confidence interval
            'left' for left tailed confidence interval
            'right' for right tailed confidence interval
        alpha : float in (0, 0.5) (default to be 0.05) 
            significance level
        
        Returns
        -------
        ci_low_est : float 
            average treatment effect (ATE) estimation
        '''
        
        ite_hat = self.dr_ite_est()
        n = len(ite_hat)
        ate_hat = np.mean(ite_hat)
        ite_sd = np.std(ite_hat)
        
        if tail == 'no':
            return ate_hat, None, None
        
        if tail == 'both': 
            z_critical = scipy.stats.norm.ppf(alpha/2) 
            ci_low_est = ate_hat + z_critical*ite_sd / np.sqrt(n)
            ci_up_est = ate_hat - z_critical*ite_sd / np.sqrt(n)
            return ate_hat, ci_low_est, ci_up_est
        
        elif tail == 'left': 
            z_critical = scipy.stats.norm.ppf(alpha) 
            ci_low_est = ate_hat + z_critical*ite_sd / np.sqrt(n)
            ci_up_est = ate_hat
            return ate_hat, ci_low_est, ci_up_est
        
        elif tail == 'right': 
            z_critical = scipy.stats.norm.ppf(alpha) 
            ci_low_est = ate_hat
            ci_up_est = ate_hat - z_critical*ite_sd / np.sqrt(n)
            return ate_hat, ci_low_est, ci_up_est
        
