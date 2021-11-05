# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 18:44:35 2021

@author: http://krasserm.github.io/2018/03/21/bayesian-optimization/
"""

from scipy.stats import norm
import numpy as np

#
# acq function
#
def expected_improvement(X, X_sample, Y_sample, gp, xi=0.01):
    '''
    Computes the EI at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model.
    
    Args:
        X: Points at which EI shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gp: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.
    
    Returns:
        Expected improvements at points X.
    '''
    mean, std = gp.predict(X, return_std=True)
    mean_sample = gp.predict(X_sample)

    std = std.reshape(-1, 1)
    
    # Needed for noise-based model,
    # otherwise use np.max(Y_sample).
    # See also section 2.4 in [1]
    mu_sample_opt = np.max(mean_sample)

    with np.errstate(divide='warn'):
        imp = mean - mu_sample_opt - xi
        Z = imp / std
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        ei[std == 0.0] = 0.0

    return ei.ravel()

def ucb(X, gp, b_n=0.01):
    if len(X.shape) == 1:
        mean, std = gp.predict(np.expand_dims(X, axis=0), return_std=True)
    else:
        mean, std = gp.predict(X, return_std=True)

    return (mean.ravel() + b_n*std.ravel())

def lcb(X, gp, b_n=0.01):
    if len(X.shape) == 1:
        mean, std = gp.predict(np.expand_dims(X, axis=0), return_std=True)
    else:
        mean, std = gp.predict(X, return_std=True)

    return (mean.ravel() - b_n*std.ravel())


from scipy.optimize import minimize

def argmax(acq_type, X_sample, Y_sample, gp, bound, bound_domain, n_restarts=25):
    '''
    Proposes the next sampling point by optimizing the acquisition function.
    
    Args:
        acq_type: Acquisition function.
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gp: A GaussianProcessRegressor fitted to samples.

    Returns:
        Location of the acquisition function maximum.
    '''
    dim = X_sample.shape[1]
    min_val = 1
    min_x = None
    
    def min_obj(X):
        # Minimization objective is the negative acquisition function
        if acq_type == 'EI':
            return -expected_improvement(X.reshape(-1, dim), X_sample, Y_sample, gp)
        elif acq_type == 'LCB':
            return -lcb(X.reshape(-1, dim), gp)
        elif acq_type == 'UCB':
            return -ucb(X.reshape(-1, dim), gp)
    

    
    
    # Find the best optimum by starting from n_restart different random points.
    # 當min_val=1時，表示困在local，重新執行
    while min_val == 1:
        # 初始化X
        X0 = np.zeros((n_restarts, dim))
        for i in range(dim):
            if bound[i]['type'] == 'continous' or bound[i]['type'] == 'fixed':
                X0[:, i] = np.random.uniform(bound_domain[i][0], bound_domain[i][1], n_restarts)
            elif bound[i]['type'] == 'discrete':
                X0[:, i] = np.random.randint(bound_domain[i][0], bound_domain[i][1], n_restarts)
        
        for x0 in X0:
            res = minimize(min_obj, x0=x0, bounds=bound_domain, method='L-BFGS-B')        
            if res.fun < min_val:
                min_val = res.fun[0]
                min_x = res.x           
            
    return min_x.reshape(-1, 1)