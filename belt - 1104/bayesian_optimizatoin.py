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
        
        # 初始化X
        X_samples = np.zeros((init_count, len(self.bound)))
        for i in range(len(self.bound)):
            if self.bound[i]['type'] == 'continous' or self.bound[i]['type'] == 'fixed':
                X_samples[:, i] = np.random.uniform(self.bound_domain[i][0], self.bound_domain[i][1], init_count)
            elif self.bound[i]['type'] == 'discrete':
                X_samples[:, i] = np.random.randint(self.bound_domain[i][0], self.bound_domain[i][1], init_count)
        
        # 初始化Y
        Y_samples = np.zeros((init_count, 1))
        for i in range(init_count):
            if unknown_func == 'score_difference':
                Y_samples[i] = unknown_function.score_difference(X_samples[i], self.train_data, self.test_data, self.test_bi_label)
            
        self.X_samples = X_samples
        self.Y_samples = Y_samples
    
    def run_native(self, n_iter, acq_type):
    
        for i in range(n_iter):
            # Update Gaussian process with existing samples
            self.gpr.fit(self.X_samples, self.Y_samples)
        
            # Obtain next sampling point from the acquisition function (expected_improvement)
            # X_next: dim * 1
            X_next = acq_func.argmax(acq_type, self.X_samples, self.Y_samples, self.gpr, self.bound, self.bound_domain)
            
            # Obtain next noisy sample from the objective function
            # Y_next: scalar
            if self.unknown_func == 'score_difference':
                Y_next = unknown_function.score_difference(X_next, self.train_data, self.test_data, self.test_bi_label)
            
            # Add sample to previous samples
            X_next = X_next.T
            self.X_samples = np.vstack((self.X_samples, X_next))
            self.Y_samples = np.vstack((self.Y_samples, Y_next))
        
        # opt result
        idx = np.argmin(self.Y_samples)
        print(-self.Y_samples[idx])
        print(self.X_samples[idx])
        return self.Y_samples[idx], self.X_samples[idx]