# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 10:50:06 2021

@author: user
"""

import matplotlib.pyplot as plt

def samples_distribution(opt_params):
    if opt_params.shape[0] == 0:
        print('opt_params is empty')
        return
    
    # (1)計算每種max_samples出現的頻率
    samples_freq = opt_params.groupby(opt_params['samples'].values).size()

    count = 0
    # (2)find 95% confidence interval
    lower_bound = 0
    upper_bound = 0
    find_lower = False
    find_upper = False
    
    for f in samples_freq.index:
        count += samples_freq[f]    
        # 找95%信賴區間的下界
        if count >= (len(opt_params) * 0.025):
            if find_lower == False:
                find_lower = True
                lower_bound = f
                print(f, samples_freq[f])
        # 找95%信賴區間的上界
        if count >= (len(opt_params) * 0.975):
            if find_upper == False:
                find_upper = True
                upper_bound = f
                print(f, samples_freq[f])

    # (3) plot
    plt.figure(figsize=(20,5))
    plt.grid(True)
    
    # plot 95% confidence interval
    plt.plot([lower_bound, lower_bound], [0, samples_freq.max()],  ':', color='r', label='95% confidence interval')
    plt.text(lower_bound, samples_freq.max(), 'samples=' + str(lower_bound), ha='center', va='top')
    plt.plot([upper_bound, upper_bound], [0, samples_freq.max()],  ':', color='r')
    plt.text(upper_bound, samples_freq.max(), 'samples=' + str(upper_bound), ha='center', va='top')
    
    # 畫 samples 分布
    plt.plot(samples_freq, 'o')
    # 標記每個點的y值
    for a, b in zip(samples_freq.index, samples_freq.values):
        plt.text(a, b, str(b), ha='left', va='top')
    
    plt.xticks(range(int(samples_freq.first_valid_index()-2), int(samples_freq.last_valid_index()), 10))
    plt.xlabel('samples')
    plt.ylabel('frequency')
    plt.title('samples distribution(' + str(opt_params.shape[0]) + ' times BO)')
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