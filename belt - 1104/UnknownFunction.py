# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 17:03:50 2021

@author: user
"""
from sklearn.ensemble import IsolationForest
import pandas as pd
import statistics


def score_difference(parameters, train_data, test_data, test_bi_label):
    IF_count = int(parameters[1])
    diff_list = []
    for i in range(IF_count):
        clf = IsolationForest(contamination=parameters[0], 
                            random_state=None,
                            n_estimators=int(parameters[2]),
                            max_samples=int(parameters[3]), 
                            max_features=int(parameters[4]))   
        clf.fit(train_data)
        
        # anomaly score
        score = clf.score_samples(test_data)        
        d = {'Anomaly_score' : pd.Series(score * -1, index=test_data.index),
             'y_test' : test_bi_label}
        df = pd.DataFrame(d)
        
        max_normal_score = df[df['y_test']==1].max(axis=0)
        min_anomaly_score = df[df['y_test']==-1].min(axis=0)
        diff_list.append(min_anomaly_score['Anomaly_score'] - max_normal_score['Anomaly_score'])
    diff_mean = statistics.mean(diff_list)
    
    return -diff_mean


def stable_tree_count(abnoraml_ratio, IF_count, tree_count, samples, features, train_data, test_data, test_bi_label):
    print('tree_count = ', tree_count)
        
    # 建立多個IF，計算多個score difference
    score_diff_list = []
    for i in range(IF_count):
        params = [abnoraml_ratio, 1, tree_count, samples, features]
        score_diff_list.append(score_difference(params, train_data, test_data, test_bi_label))

    # 計算最好、最差的score differenc之間的落差
    score_min = -max(score_diff_list)
    score_max = -min(score_diff_list)
    diff_presentage = (score_max - score_min) / score_max
    return score_min, score_max, diff_presentage