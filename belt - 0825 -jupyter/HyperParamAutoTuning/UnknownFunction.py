# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 16:44:07 2021

@author: user
"""

from sklearn.ensemble import IsolationForest
import pandas as pd


class UnknownFunc():
    def __init__(self, parameters, X_train, X_test, y_test_bi):
        self.X_train = X_train
        self.X_test = X_test
        self.y_test_bi = y_test_bi
        self.parameters = parameters

    def IF_1(self):
        IF_count = 1
        diff_list = []
        for i in range(IF_count):
            clf = IsolationForest(contamination=self.parameters[0], 
                                n_estimators=int(self.parameters[1]), 
                                random_state=None,
                                max_samples=int(self.parameters[2]), 
                                max_features=int(self.parameters[3]))   
            clf.fit(self.X_train)
            
            # anomaly score
            score = clf.score_samples(self.X_test)
            
            d = {'Anomaly_score' : pd.Series(score * -1, index=self.X_test.index),
                 'y_test' : self.y_test_bi}
            df = pd.DataFrame(d)
            
            max_normal_score = df[df['y_test']==1].max(axis=0)
            min_anomaly_score = df[df['y_test']==-1].min(axis=0)
            diff_list.append(min_anomaly_score['Anomaly_score'] - max_normal_score['Anomaly_score'])
        diff_min = min(diff_list)
        print(diff_min)
        
        return -diff_min