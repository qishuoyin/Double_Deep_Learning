#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:01:48 2024

@author: Qishuo

FAST_NN_ATE: model structure -- Vanilla NN

"""

import numpy as np
import torch
from torch import nn
from collections import OrderedDict


# run script on gpu if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# network structure
# base Vanilla network structure
class VanillaNetBase(nn.Module):
    
    '''
    A class for Vanilla neural network base model

    Attributes
    ----------
    linear_relu_stack: nn.module
        the relu nerual network module
    sigmoid: nn.module
        the sigmoid module

    Methods
    -------
    __init__(): initilize the module
    forward(x): implementation of the forward pass
    least_square_loss: compute the least square loss
    cross_entropy_loss: compute the cross entropy loss
    '''
    
    def __init__(self, input_dim, depth, width, logistic=False):
        
        '''
        Parameters
        ----------
        input_dim: int
		    the number of input features
        depth: int
		    the number of hidden layers of neural network
		width: int
		    the number of hidden units in each layer
        logistic: bool
		    whether to use logistic activation function
        '''
        
        super(VanillaNetBase, self).__init__()
        self.input_dim = input_dim
        self.depth = depth
        self.width = width
        self.logistic = logistic

        relu_nn = [('Linear1', nn.Linear(input_dim, width)), ('ReLU1', nn.ReLU())]
        for i in range(depth-1):
            relu_nn.append(('Linear' + str(i+2), nn.Linear(width, width)))
            relu_nn.append(('ReLU' + str(i+2), nn.ReLU()))
        relu_nn.append(('linear{}'.format(depth+1), nn.Linear(width, 1)))

        self.linear_relu_stack = nn.Sequential(
            OrderedDict(relu_nn)
        )

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        
        '''
        A function to implement forward pass
    
        ...
        Parameters
        ----------
        x: torch.tensor
            (n, input_dim) matrix of the input

        Returns
        -------
        pred: torch.tensor
            (n, 1) matrix of prediction
        '''
        
        pred = self.linear_relu_stack(x)
        if self.logistic == True:
            pred = self.sigmoid(pred)
        return pred

    def least_square_loss(self):
        
        '''
        A function to compute the least square loss
        
        ...
        Returns
        -------
        loss: torch.tensor
            a scalar of the least square loss
        '''
        
        loss = nn.MSELoss()
        return loss

    def cross_entropy_loss(self):
        
        '''
        A function to compute the cross entropy loss
        
        ...
        Returns
        -------
        loss: torch.tensor
            a scalar of the corss entropy loss
        '''
        
        loss = nn.CrossEntropyLoss()
        return loss
