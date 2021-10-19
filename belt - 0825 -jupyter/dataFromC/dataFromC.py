# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 20:26:14 2021

@author: user
"""
import os
import pandas as pd

def DataFromC():
    path = os.path.abspath(os.getcwd()) + '\\dataFromC\\anomaly_scores.csv'
    return pd.read_csv(path, names=['anomaly_score', 'average_path_length', 'c_value'], index_col=False)