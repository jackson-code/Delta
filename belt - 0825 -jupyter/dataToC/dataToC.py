# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 20:14:22 2021

@author: user
"""

import os

def DataToC(train_X, test_Xy):
    path = os.path.abspath(os.getcwd()) + '\\dataToC\\'

    file_name = 'train_data.csv'
    train_X.to_csv(path_or_buf=path + file_name, index=False, header=False)
    
    file_name = 'test_data.csv'
    test_Xy.to_csv(path_or_buf=path + file_name, index=False, header=False)