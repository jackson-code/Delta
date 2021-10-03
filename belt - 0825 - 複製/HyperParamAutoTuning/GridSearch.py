# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 16:36:09 2021

@author: user
"""

import time
import pandas as pd
from .UnknownFunction import UnknownFunc


class GridSearch():
    def __init__(self, bound_n_estimators, bound_max_samples, bound_max_features):
        self.bound_n_estimators = bound_n_estimators
        self.bound_max_samples = bound_max_samples
        self.bound_max_features = bound_max_features
        result = pd.DataFrame(columns=['n_estimators', 'max_samples', 'max_features', 'score_difference'])
    
    def run(self):
        time_start = time.time() #開始計時
        for n_estimators in self.bound_n_estimators:
            for max_samples in self.bound_max_samples:
                for max_features in self.bound_max_features:
                    params = [n_estimators, max_samples, max_features]
                    UnknownFunc
                    score_difference = (params)
                    result = result.append({
                        'n_estimators': n_estimators,
                        'max_samples': max_samples, 
                        'max_features': max_features,
                        'score_difference': score_difference}, ignore_index=True)
        time_end = time.time()    #結束計時