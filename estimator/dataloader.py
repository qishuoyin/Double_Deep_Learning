#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 16:36:51 2024

@author: Qishuo

Double Deep Learning: dataloader

"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# dataloader
class data_to_torch(Dataset):
    
    '''
    A class to transform numpy data to torch data
    
    ...
    Methods
    -------
    __init__()
        initialize the module
    __len__()
        measure the dataset size -- n
    __getitem__()
        transform x, y from numpy to torch
    '''
    
    def __init__(self, x, y):
        
        '''
        Parameters
        ----------
        n : int
            size of the dataset
        feature : numpy.ndarray
            (n, p) matrix of input feature data
        response : numpy.ndarray
            (n, ) matrix of input target data
        '''
        
        self.n = np.shape(x)[0]
        self.feature = x
        self.response = y


    def __len__(self):
        
        '''
        Returns
        -------
        n : int
            size of the dataset
        '''
        
        return self.n


    def __getitem__(self, idx):
        
        '''
        Parameters
        ----------
        idx : int
            index of feature and target

        Returns
        -------
        torch tensor pair of feature and target
        '''
        
        return torch.tensor(self.feature[idx, :], dtype=torch.float32), torch.tensor(self.response[idx, :], dtype=torch.float32)


def data_torch_dataloader(data_torch, batch_size):
    
    '''
    A function to convert torch tensorn data to dataloader
    
    ...
    Parameters
    ----------
    data_torch : (torch.tensor, torch.tensor)
        torch data of feature and target
    batch_size : int
        batch size -- number of observations put into each loader

    Returns
    -------
    dataloader : torch.utils.data.dataloader.DataLoader
        dataloader of given torch tensor pair (feature_torch, target_torch)
    '''
    
    dataloader = DataLoader(data_torch, batch_size=batch_size, shuffle=True)
    return dataloader
