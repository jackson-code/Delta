# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 15:39:07 2021

@author: user
"""

def binary_search_dynamic_high(low, high, stop_interval, threshold, abnormal_ratio, samples, features, stable_tree_count_func, train_data, test_data, test_bi_label):
    stable_n_estimators = []
    all_n_estimators = []
    diff_presentage_list = []
    stable_diff_presentage_list = []    
    # 決定high bound
    score_min, score_max, score_diff = stable_tree_count_func(
        abnormal_ratio, 30, high, samples, features, 
        train_data, test_data, test_bi_label)
    
    if score_diff < (threshold*0.9):
        stable_n_estimators.append(high)
        stable_diff_presentage_list.append(score_diff)
        all_n_estimators.append(high)
        diff_presentage_list.append(score_diff)       
    else:
        while score_diff > (threshold*0.9):
            low = high
            high = high * 2
            score_min, score_max, score_diff = stable_tree_count_func(
                abnormal_ratio, 30, high, samples, features, 
                train_data, test_data, test_bi_label)
            
            all_n_estimators.append(high)
            diff_presentage_list.append(score_diff)
    stable_n_estimators.append(high)
    stable_diff_presentage_list.append(score_diff)
    

    # binary search
    while (high-low) >= stop_interval:         
        mid = (high + low) // 2     # //: floor
        print('mid =', mid)
        all_n_estimators.append(mid)
        
        score_min, score_max, score_diff = stable_tree_count_func(
            abnormal_ratio, 30, mid, samples, features,
            train_data, test_data, test_bi_label)
        diff_presentage_list.append(score_diff)

        if score_diff <= threshold:
            stable_n_estimators.append(mid)
            stable_diff_presentage_list.append(score_diff)
            high = mid # 往左搜尋
            mid = (high + low) // 2 
        else: 
            low = mid
            mid = (high + low) // 2
    
    # 最小的穩定n_estimator
    stable_n_estimator = min(stable_n_estimators)
    # 與最小的穩定n_estimator
    stable_diff_presentage = stable_diff_presentage_list[stable_n_estimators.index(stable_n_estimator)]
    
    return diff_presentage_list, all_n_estimators, stable_n_estimator, stable_diff_presentage

# 原始的二元搜尋，已棄用
def binary_search(low, high, stop_interval, threshold, samples, features, stable_tree_count_func):
    upper_bound = high
    stable_n_estimators = []
    all_n_estimators = []
    diff_presentage_list = []
    while (high-low) >= stop_interval:         
        mid = (high + low) // 2     # //: floor
        print('mid =', mid)
        all_n_estimators.append(mid)
        
        score_min, score_max, score_diff = stable_tree_count_func(
            IF_count=30, n_estimators=mid, 
            samples=samples, features=features)
        diff_presentage_list.append(score_diff)

        if score_diff <= threshold:
            stable_n_estimators.append(mid)
            high = mid # 往左搜尋
            mid = (high + low) // 2 
        else: 
            low = mid
            mid = (high + low) // 2
            
    if len(stable_n_estimators) > 0:        
        return diff_presentage_list, all_n_estimators, min(stable_n_estimators)
    else:
        return diff_presentage_list, all_n_estimators, upper_bound