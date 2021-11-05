# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 17:01:09 2021

@author: user
"""
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
import numpy as np

import AcquisitionFunction as acq_func
import UnknownFunction 


# Gaussian process with Mate'rn kernel as surrogate model
m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
gpr = GaussianProcessRegressor(kernel=m52)

def BO_run(n_iter, X_sample ,Y_sample, bound, bound_domain, train_data, test_data, test_label_bi):

    for i in range(n_iter):
        # Update Gaussian process with existing samples
        gpr.fit(X_sample, Y_sample)
    
        # Obtain next sampling point from the acquisition function (expected_improvement)
        # X_next: dim * 1
        X_next = acq_func.argmax('LCB', X_sample, Y_sample, gpr, bound_domain)
        print(X_next)
        # 得到的argmax(acq)會是continous，必須把discrete type的數值取floor 
        for j in range(len(bound)):
            if bound[j]['type'] == 'discrete':
                X_next[j] = np.floor(X_next[j])
        
        # Obtain next noisy sample from the objective function
        # Y_next: scalar
        Y_next = UnknownFunction.score_difference(X_next, train_data, test_data, test_label_bi)
        
        # Add sample to previous samples
        X_next = X_next.T
        X_sample = np.vstack((X_sample, X_next))
        Y_sample = np.vstack((Y_sample, Y_next))
    
    # opt result
    idx = np.argmin(Y_sample)
    print(-Y_sample[idx])
    print(X_sample[idx])
    return Y_sample[idx], X_sample[idx]