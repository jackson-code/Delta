# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 17:03:50 2021

@author: user
"""
from sklearn.ensemble import IsolationForest
import pandas as pd


def score_difference(parameters, X_train, X_test, y_test_bi):
    IF_count = int(parameters[1])
    diff_list = []
    for i in range(IF_count):
        clf = IsolationForest(contamination=parameters[0], 
                            random_state=None,
                            n_estimators=int(parameters[2]),
                            max_samples=int(parameters[3]), 
                            max_features=int(parameters[4]))   
        clf.fit(X_train)
        
        # anomaly score
        score = clf.score_samples(X_test)        
        d = {'Anomaly_score' : pd.Series(score * -1, index=X_test.index),
             'y_test' : y_test_bi}
        df = pd.DataFrame(d)
        
        max_normal_score = df[df['y_test']==1].max(axis=0)
        min_anomaly_score = df[df['y_test']==-1].min(axis=0)
        diff_list.append(min_anomaly_score['Anomaly_score'] - max_normal_score['Anomaly_score'])
    diff_min = min(diff_list)
    # diff_max = max(diff_list)
    #print(diff_min)
    
    return -diff_min