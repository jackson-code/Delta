# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 20:10:06 2021

@author: user
"""
import PROPERTY

def remove_partial_abnormal_data(train_data_label, abnormal_ratio, normal_span, abnormal_span):
    abnormal_ratio = abnormal_ratio / len(abnormal_span)
    print('\t processing Train data')
    for i in range(1, PROPERTY.Experiment().SPAN_COUNT+1) :
        if i in abnormal_span:
            print('\t add partial train abnormal data')
            # 取全部異常資料
            partial_train_label = train_data_label.loc[train_data_label.Span == i, :]
            # 依異常比例取部分的 train 異常資料
            partial_train_label = partial_train_label.sample(frac = abnormal_ratio, axis = 0)
            
            # 移除所有的異常資料
            idx = train_data_label.loc[train_data_label.Span == i,:].index
            train_data_label = train_data_label.drop(idx, axis = 0)
            # 加上部分的異常資料
            train_data_label = train_data_label.append(partial_train_label)
        # 正常資料不處理
        elif i in normal_span:
            continue
        # 移除不用的正常資料(不在normal_span的正常資料會被移除)
        else: 
            idx = train_data_label.loc[train_data_label.Span == i,:].index
            train_data_label = train_data_label.drop(idx, axis = 0)
    
    train_data = train_data_label.drop(columns='Span')
    train_label = train_data_label.Span
    return train_data, train_label