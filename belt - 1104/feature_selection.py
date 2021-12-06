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
        Sorted square Pearson coefficient
        Pearson = -1~1

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

def spearman(x, y):
    '''
    Reference:
        https://www.youtube.com/watch?v=Zc9sm1irUx8&ab_channel=CUSTCourses
        
    適用於非線性
    
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
        Sorted square Spearman coefficient
        Spearman = -1~1
    '''
    rank_x = x.rank()
    rank_y = y.rank()
    # 計算每種feature的spearman
    spearman_list = []
    for i in range(rank_x.shape[1]):
        d = rank_x.iloc[:, i] - rank_y
        d = d**2
        # 分子
        numerator = 6 * d.sum()
        # 分母
        n = rank_x.shape[0]
        denominator = n * (n**2 - 1)
        
        spearman = 1 - (numerator / denominator)
        # 原本的spearman係數取平方
        spearman_list.append(spearman**2) 
        
    spearman = pd.Series(spearman_list, index=x.columns)
    spearman.sort_values(inplace=True)
    return spearman