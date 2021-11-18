# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 17:01:09 2021

@author: user
"""
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
import numpy as np

import acquisition_function as acq_func
import unknown_function 
import process_data

class BayesianOptimization():
    def __init__(self, bound, train_data, test_data, test_bi_label):
        self.bound = bound
        bound_domain = np.zeros((len(bound), 2))
        for i in range(len(bound)):
            if bound[i]['type'] == 'fixed':
                bound_domain[i] = [ bound[i]['domain'], bound[i]['domain']]
            else:
                bound_domain[i] = [ bound[i]['domain'].start, bound[i]['domain'].stop ]
        self.bound_domain = bound_domain
        
        self.train_data = train_data
        self.test_data = test_data
        self.test_bi_label = test_bi_label
        
        # Gaussian process with Mate'rn kernel as surrogate model
        m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        self.gpr = GaussianProcessRegressor(kernel=m52)
        
    def init_samples(self, init_count, unknown_func):
        self.unknown_func = unknown_func
        
        # 初始化 X，還沒scale
        X_original = np.zeros((init_count, len(self.bound)))
        for i in range(len(self.bound)):
            if self.bound[i]['type'] == 'continous' or self.bound[i]['type'] == 'fixed':
                X_original[:, i] = np.random.uniform(self.bound_domain[i][0], self.bound_domain[i][1], init_count)
            elif self.bound[i]['type'] == 'discrete':
                X_original[:, i] = np.random.randint(self.bound_domain[i][0], self.bound_domain[i][1], init_count)
        
        # 初始化Y，還沒scale
        Y_original = np.zeros((init_count, 1))
        for i in range(init_count):
            if unknown_func == 'score_difference':
                Y_original[i] = unknown_function.score_difference(X_original[i], self.train_data, self.test_data, self.test_bi_label)
            
        self.X_original = X_original
        self.Y_original = Y_original
        

    
    def run_native(self, n_iter, acq_type):
    
        for i in range(n_iter):
            # 把重複的row去除，for sklearn gp(成功消除了gp的警告訊息)
            self.X_original, unique_idx = np.unique(self.X_original, axis=0, return_index=True)
            self.Y_original = self.Y_original[unique_idx]
            
            # scale for sklearn gp (不確定需不需要，paper code都有且x也有scale，x先暫時不scale)
            self.Y_scaled = (self.Y_original - np.mean(self.Y_original)) / (np.max(self.Y_original) - np.min(self.Y_original))
        
            # Update Gaussian process with existing samples
            self.gpr.fit(self.X_original, self.Y_scaled)
        
            # Obtain next sampling point from the acquisition function (expected_improvement)
            # X_next: dim * 1
            X_next = acq_func.argmax(acq_type, self.X_original, self.Y_scaled, self.gpr, self.bound, self.bound_domain)
            
            # Obtain next noisy sample from the objective function
            # Y_next: scalar
            if self.unknown_func == 'score_difference':
                Y_next = unknown_function.score_difference(X_next, self.train_data, self.test_data, self.test_bi_label)
            
            # Add sample to previous samples
            X_next = X_next.T
            self.X_original = np.vstack((self.X_original, X_next))
            self.Y_original = np.vstack((self.Y_original, Y_next))
        
        # opt result
        idx = np.argmin(self.Y_original)
        print(-self.Y_original[idx])
        print(self.X_original[idx])
        return self.Y_original[idx], self.X_original[idx]
    
    def run_abnormal_ratio(self, n_iter, acq_type, train_data_label):
        normal_span = [1]
        abnormal_span = [2]
        
        for i in range(n_iter):
            # Update Gaussian process with existing samples
            self.gpr.fit(self.X_original, self.Y_original)
        
            # Obtain next sampling point from the acquisition function (expected_improvement)
            # X_next: dim * 1
            X_next = acq_func.argmax(acq_type, self.X_original, self.Y_original, self.gpr, self.bound, self.bound_domain)
            
            # X_next[0] is abnormal ratio
            # 依abnormal ratio調整train data
            train_data, train_label = process_data.remove_partial_abnormal_data(train_data_label, float(X_next[0]), 
                                                                        normal_span, abnormal_span)
            
            # Obtain next noisy sample from the objective function
            # Y_next: scalar
            if self.unknown_func == 'score_difference':
                Y_next = unknown_function.score_difference(X_next, train_data, self.test_data, self.test_bi_label)
            
            # Add sample to previous samples
            X_next = X_next.T
            self.X_original = np.vstack((self.X_original, X_next))
            self.Y_original = np.vstack((self.Y_original, Y_next))
        
        # opt result
        idx = np.argmin(self.Y_original)
        print(-self.Y_original[idx])
        print(self.X_original[idx])
        return self.Y_original[idx], self.X_original[idx]