# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 11:00:00 2021

@author: user
"""
import pandas as pd

def pearson(x, y):
    '''
    Reference Paper: An introduction to variable and feature selection(Isabelle Guyon)
        section 2.2
    適用於線性
    
    Parameters
    ----------
    x : m*n DataFrame
        m is # of data(examples)
        n is # of features
    y : m*1 array
        output of the model(target)
    Returns
    -------
    m*1 Series
        Sorted Pearson coefficient

    '''
    y_term = y - y.mean()
    pearson_list = []
    # 計算每種feature的pearson^2
    for i in range(x.shape[1]):    
        x_term = x.iloc[:, i] - x.iloc[:, i].mean()
        # 分子
        numerator = (x_term * y_term).sum()
        # 分母
        denominator = ((x_term ** 2).sum() * (y_term ** 2).sum()) ** (1/2)
        # 公式(2)平方
        pearson_list.append((numerator / denominator)**2)
    
    pearson = pd.Series(pearson_list, index=x.columns)
    pearson.sort_values(inplace=True)
    return pearson