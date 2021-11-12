# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 10:50:06 2021

@author: user
"""

import matplotlib.pyplot as plt

def samples_distribution(opt_samples):
    if opt_samples.shape[0] == 0:
        print('opt_samples is empty')
        return
    
    # (1)計算每種max_samples出現的頻率
    # samples_freq = opt_params.groupby(opt_params['samples'].values).size()
    samples_freq = opt_samples.value_counts()
    
    std = opt_samples.std()
    mean = opt_samples.mean()

    # (3) plot
    plt.figure(figsize=(20,5))
    plt.grid(True)
 
    # mean
    mean = round(mean, 2)
    plt.plot([mean, mean], [0, samples_freq.max()],  ':', color='b', label='mean')
    plt.text(mean, samples_freq.max(), str(mean), ha='center', va='top')
    
    # plot 68% confidence interval
    lower_bound1 = round(mean - std, 2)
    plt.plot([lower_bound1, lower_bound1], [0, samples_freq.max()],  ':', color='r', label='68% confidence interval')
    plt.text(lower_bound1, samples_freq.max(), str(lower_bound1), ha='center', va='top')
    upper_bound1 = round(mean + std, 2)
    plt.plot([upper_bound1, upper_bound1], [0, samples_freq.max()],  ':', color='r')
    plt.text(upper_bound1, samples_freq.max(), str(upper_bound1), ha='center', va='top')
    
    # plot 95% confidence interval
    lower_bound2 = round(mean - 2*std, 2)
    plt.plot([lower_bound2, lower_bound2], [0, samples_freq.max()],  ':', color='g', label='95% confidence interval')
    plt.text(lower_bound2, samples_freq.max(), str(lower_bound2), ha='center', va='top')
    upper_bound2 = round(mean + 2*std, 2)
    plt.plot([upper_bound2, upper_bound2], [0, samples_freq.max()],  ':', color='g')
    plt.text(upper_bound2, samples_freq.max(), str(upper_bound2), ha='center', va='top')
    
    # 畫 samples 分布
    plt.plot(samples_freq, 'o')
    # 標記每個點的y值
    for a, b in zip(samples_freq.index, samples_freq.values):
        plt.text(a, b, str(b), ha='left', va='top')
    
    plt.xticks(range(int(samples_freq.first_valid_index()-2), int(samples_freq.last_valid_index()), 10))
    plt.xlabel('samples')
    plt.ylabel('frequency')
    plt.title('samples distribution(' + str(opt_samples.shape[0]) + ' times BO)')
    plt.legend()

def features_distribution(opt_params):
    if opt_params.shape[0] == 0:
        print('opt_params is empty')
        return
    
    plt.figure(figsize=(7,5))
    plt.grid(True)
    
    features_freq = opt_params.groupby(opt_params['features'].values).size() 
    plt.plot(features_freq, 'o')
    
    for a, b in zip(features_freq.index, features_freq.values):
        plt.text(a, b, str(b), ha='left', va='top')
    
    plt.xlabel('features')
    plt.ylabel('frequency')
    plt.title('features distribution(' + str(opt_params.shape[0]) + ' times BO)')