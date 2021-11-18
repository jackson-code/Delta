# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 11:54:21 2021

@author: user
"""
import numpy as np
from scipy.sparse import issparse
from sklearn.utils.extmath import row_norms, safe_sparse_dot

def r_regression(X, y, *, center=True):
    """Compute Pearson's r for each features and the target.
    Pearson's r is also known as the Pearson correlation coefficient.
    .. versionadded:: 1.0
    Linear model for testing the individual effect of each of many regressors.
    This is a scoring function to be used in a feature selection procedure, not
    a free standing feature selection procedure.
    The cross correlation between each regressor and the target is computed
    as ((X[:, i] - mean(X[:, i])) * (y - mean_y)) / (std(X[:, i]) * std(y)).
    For more on usage see the :ref:`User Guide <univariate_feature_selection>`.
    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data matrix.
    y : array-like of shape (n_samples,)
        The target vector.
    center : bool, default=True
        Whether or not to center the data matrix `X` and the target vector `y`.
        By default, `X` and `y` will be centered.
    Returns
    -------
    correlation_coefficient : ndarray of shape (n_features,)
        Pearson's R correlation coefficients of features.
    See Also
    --------
    f_regression: Univariate linear regression tests returning f-statistic
        and p-values
    mutual_info_regression: Mutual information for a continuous target.
    f_classif: ANOVA F-value between label/feature for classification tasks.
    chi2: Chi-squared stats of non-negative features for classification tasks.
    """
    n_samples = X.shape[0]

    # Compute centered values
    # Note that E[(x - mean(x))*(y - mean(y))] = E[x*(y - mean(y))], so we
    # need not center X
    if center:
        y = y - np.mean(y)
        if issparse(X):
            X_means = X.mean(axis=0).getA1()
        else:
            X_means = X.mean(axis=0)
        # Compute the scaled standard deviations via moments
        X_norms = np.sqrt(row_norms(X.T, squared=True) - n_samples * X_means ** 2)
    else:
        X_norms = row_norms(X.T)

    correlation_coefficient = safe_sparse_dot(y, X)
    correlation_coefficient /= X_norms
    correlation_coefficient /= np.linalg.norm(y)
    return correlation_coefficient


def r_regression(X, y):
    """
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data matrix.
    y : array-like of shape (n_samples,)
        The target vector.
    Returns
    -------
    correlation_coefficient : ndarray of shape (n_features,)
        Pearson's R correlation coefficients of features.
    """
    n_samples = X.shape[0]

    # Compute centered values
    # Note that E[(x - mean(x))*(y - mean(y))] = E[x*(y - mean(y))], so we
    # need not center X

    y = y - np.mean(y)
    X_means = X.mean(axis=0)
    X_std = X.std()
    y_std = y.std()


    correlation_coefficient = safe_sparse_dot(y, X)
    correlation_coefficient /= X_norms
    correlation_coefficient /= np.linalg.norm(y)
    return correlation_coefficient