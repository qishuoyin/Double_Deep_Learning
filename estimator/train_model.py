#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:07:56 2024

@author: Qishuo

Double Deep Learning: training model

"""

import torch
import matplotlib.pyplot as plt 


# run script on gpu if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NNModelTrain:
    
    '''
    A class to train a NN model 
    
    ...
    
    Methods
    -------
    __init__()
        initialize the module
    train(dataloader, logistic, regularization_type=None, lambda_reg=1, penalty_weight=None, reg_tau=0.005)
        train NN model for a single epoch
    test(dataloader, logistic, regularization_type=None, lambda_reg=1, penalty_weight=None, reg_tau=0.005)
        test NN model for a single epoch
    fit(dataloader, logistic, regularization_type=None, lambda_reg=1, penalty_weight=None, tau0=0.005)
        fit the NN model by given dataloader for multiple epochs
    predict(fitted_model, x)
        predict the target on the given feature by the fitted model
    '''
    
    def __init__(self,
                 epochs,
                 batchsize,
                 model,
                 loss_fn,
                 optimizer):
        
        '''
        Parameters
        ----------
        epochs : int
            numbers of epochs in training & test process
        batchsize : int
            batch size -- number of observations put into each loader
        model : NN model class
        loss_fn : loss function defined in NN model class
        optimizer : optmizer used in the training & test process
        '''

        self.epochs = epochs
        self.batchsize = batchsize
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer


    def train(self, dataloader, logistic, regularization_type=None, lambda_reg=1, penalty_weight=None, reg_tau=0.005):
        
        '''
        A function to train NN model for a single epoch
        
        ...
        Parameters
        ----------
        dataloader : torch.utils.data.dataloader.DataLoader
            dataloader of given torch tensor pair (feature_torch, target_torch)
        logistic : bool
            whether the model is a classification
            'True' for classification model
            'False' for regression model
        regularization_type : NoneType or str
            'None' for no regularization in Vanilla NN model
            'L1' for L1 regularization in Vanilla NN model
            'L2' for L2 regularization in Vanilla NN model
        lambda_reg : float (default to be 1)
            regularization parameter in Vanilla NN model if regularization_type is not None
        penalty_weight : NoneType or float
            'None' for no penalty in FAST NN model
            float for the penalty value in FAST NN model
        reg_tau : float (default to be 0.005)
            clipping threshold in FAST NN model if penalty_weight is not None
            
        Returns
        -------
        train_err : float
            training error for a single epoch
        '''
        
        model = self.model
        loss_fn = self.loss_fn
        optimizer = self.optimizer

        model.train()
        model = model.to(device)
        num_batches = len(dataloader)
        train_loss_sum = 0

        for batch, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            train_loss = loss_fn(pred, y)

            # apply L1 regularization
            if regularization_type == 'L1':
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                train_loss += lambda_reg * l1_norm

            # apply L2 regularization
            elif regularization_type == 'L2':
                l2_norm = sum(p.pow(2).sum() for p in model.parameters())
                train_loss += lambda_reg * l2_norm

            if penalty_weight is not None:
                reg_loss = model.regularization_loss(reg_tau)
                train_loss += penalty_weight * reg_loss

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_loss_sum += train_loss.item()

        train_err = train_loss_sum / num_batches
        return train_err


    # test function just in case
    def test(self, dataloader, logistic, regularization_type=None, lambda_reg=1, penalty_weight=None, reg_tau=0.005):
        
        '''
        A function to test NN model for a single epoch
        
        ...
        Parameters
        ----------
        dataloader : torch.utils.data.dataloader.DataLoader
            dataloader of given torch tensor pair (feature_torch, target_torch)
        logistic : bool
            whether the model is a classification
            'True' for classification model
            'False' for regression model
        regularization_type : NoneType or str
            'None' for no regularization in Vanilla NN model
            'L1' for L1 regularization in Vanilla NN model
            'L2' for L2 regularization in Vanilla NN model
        lambda_reg : float (default to be 1)
            regularization parameter in Vanilla NN model if regularization_type is not None
        penalty_weight : NoneType or float
            'None' for no penalty in FAST NN model
            float for the penalty value in FAST NN model
        reg_tau : float (default to be 0.005)
            clipping threshold in FAST NN model if penalty_weight is not None
            
        Returns
        -------
        test_err : float
            test error for a single epoch
        '''
        
        model = self.model
        loss_fn = self.loss_fn
        optimizer = self.optimizer

        model.eval()
        model = model.to(device)
        num_batches = len(dataloader)
        test_loss_sum = 0

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                test_loss = loss_fn(pred, y)

                # apply L1 regularization
                if regularization_type == 'L1':
                    l1_norm = sum(p.abs().sum() for p in model.parameters())
                    test_loss += lambda_reg * l1_norm

                # apply L2 regularization
                elif regularization_type == 'L2':
                    l2_norm = sum(p.pow(2).sum() for p in model.parameters())
                    test_loss += lambda_reg * l2_norm
                
                # apply clipping regularization
                if penalty_weight is not None:
                    reg_loss = model.regularization_loss(reg_tau)
                    test_loss += penalty_weight * reg_loss

                test_loss_sum += test_loss.item()

        test_err = test_loss_sum / num_batches
        return test_err


    # def fit(self, dataloader_train, logistic, regularization_type=None, lambda_reg=1, penalty_weight=None, tau0=0.005, dataloader_test=None):
        
    #     '''
    #     A function to train model on a given dataloader for a series of epoches
        
    #     ...
    #     Parameters
    #     ----------
    #     dataloader_train : torch.utils.data.dataloader.DataLoader
    #         dataloader of training set
    #     logistic : bool
    #         whether the model is a classification
    #         'True' for classification model
    #         'False' for regression model
    #     regularization_type : NoneType or str
    #         'None' for no regularization in Vanilla NN model
    #         'L1' for L1 regularization in Vanilla NN model
    #         'L2' for L2 regularization in Vanilla NN model
    #     lambda_reg : float (default to be 1)
    #         regularization parameter in Vanilla NN model if regularization_type is not None
    #     penalty_weight : NoneType or float
    #         'None' for no penalty in FAST NN model
    #         float for the penalty value in FAST NN model
    #     reg_tau : float (default to be 0.005)
    #         clipping threshold in FAST NN model if penalty_weight is not None
    #     dataloader_train : torch.utils.data.dataloader.DataLoader (default to be None)
    #          dataloader of test set
    #     '''
        
    #     anneal_rate = (tau0 * 20 - tau0) / self.epochs
    #     anneal_tau = tau0 * 20
        
    #     train_err_vec = []
        
    #     for epoch in range(self.epochs):
    #         print(f"Epoch {epoch+1}")
            
    #         # apply L1 regularization
    #         if regularization_type == 'L1':
    #             train_err = self.train(dataloader_train, logistic, regularization_type='L1', lambda_reg=1)

    #         # apply L2 regularization
    #         elif regularization_type == 'L2':
    #             train_err = self.train(dataloader_train, logistic, regularization_type='L2', lambda_reg=1)
            
    #         # apply clipping regularization
    #         if penalty_weight is not None:
    #             anneal_tau -= anneal_rate
    #             train_err = self.train(dataloader_train, logistic, penalty_weight=penalty_weight, reg_tau=anneal_tau)
    #             #print("reg_tau:" + str(anneal_tau))
                
    #         if regularization_type == None and penalty_weight == None: 
    #             train_err = self.train(dataloader_train, logistic)
            
    #         print(f"train error = {train_err}")
    #         print("-------------------------------")
    #         train_err_vec.append(train_err)
            
        
    #     # plt.figure(figsize=(16, 8))
    #     # plt.plot(train_err_vec, color='red')
    #     # plt.savefig(f"train loss.pdf")
    #     # plt.close()
        
    #     return self.model
    
    
    def fit(self, dataloader_train, dataloader_val, logistic, regularization_type=None, lambda_reg=1, penalty_weight=None, tau0=0.005, dataloader_test=None):
        
        '''
        A function to train model on a given dataloader for a series of epoches
        
        ...
        Parameters
        ----------
        dataloader_train : torch.utils.data.dataloader.DataLoader
            dataloader of training set
        dataloader_val : torch.utils.data.dataloader.DataLoader
            dataloader of validation set
        logistic : bool
            whether the model is a classification
            'True' for classification model
            'False' for regression model
        regularization_type : NoneType or str
            'None' for no regularization in Vanilla NN model
            'L1' for L1 regularization in Vanilla NN model
            'L2' for L2 regularization in Vanilla NN model
        lambda_reg : float (default to be 1)
            regularization parameter in Vanilla NN model if regularization_type is not None
        penalty_weight : NoneType or float
            'None' for no penalty in FAST NN model
            float for the penalty value in FAST NN model
        reg_tau : float (default to be 0.005)
            clipping threshold in FAST NN model if penalty_weight is not None
        dataloader_train : torch.utils.data.dataloader.DataLoader (default to be None)
             dataloader of test set
        '''
        
        anneal_rate = (tau0 * 20 - tau0) / self.epochs
        anneal_tau = tau0 * 20
        
        train_err_vec = []
        val_err_vec = []
        
        # early stopping parameters
        patience = 10 # Number of epochs to wait before stopping
        min_delta = 0 # Minimum change in loss to qualify as an improvement
        patience_counter = 0
        best_err = float('inf')
        
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}")
            
            # apply L1 regularization
            if regularization_type == 'L1':
                train_err = self.train(dataloader_train, logistic, regularization_type='L1', lambda_reg=1)
                val_err = self.test(dataloader_val, logistic, regularization_type='L1', lambda_reg=1)

            # apply L2 regularization
            elif regularization_type == 'L2':
                train_err = self.train(dataloader_train, logistic, regularization_type='L2', lambda_reg=1)
                val_err = self.test(dataloader_val, logistic, regularization_type='L2', lambda_reg=1)
            
            # apply clipping regularization
            if penalty_weight is not None:
                anneal_tau -= anneal_rate
                train_err = self.train(dataloader_train, logistic, penalty_weight=penalty_weight, reg_tau=anneal_tau)
                val_err = self.test(dataloader_val, logistic, penalty_weight=penalty_weight, reg_tau=anneal_tau)
                #print("reg_tau:" + str(anneal_tau))
                
            if regularization_type == None and penalty_weight == None: 
                train_err = self.train(dataloader_train, logistic)
                val_err = self.test(dataloader_val, logistic)
            
            print(f"train error = {train_err}")
            print(f"validation error = {val_err}")
            print("-------------------------------")
            train_err_vec.append(train_err)
            val_err_vec.append(val_err)
            
            # implement early stopping
            if val_err < best_err - min_delta:
                best_err = val_err
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        return self.model


    def predict(self, fitted_model, x):
        
        '''
        
        A function to predict the target on the given feature by the fitted model

        Parameters
        ----------
        fitted_model : NN model
            fitted model in 'fit' function
        x : numpy.ndarray
            (n, p) matrix of input feature data for prediction

        Returns
        -------
        pred : torch tensor
            predicted target given feature x
        '''
        
        fitted_model.eval()
        fitted_model = fitted_model.to(device)
        x = torch.Tensor(x)
        x = x.to(device)
        with torch.no_grad():
          pred = fitted_model(x)
        return pred
    
