# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 21:27:20 2021

@author: user
"""

#%%
import os
import pandas as pd
import math
import glob
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.model_selection import train_test_split
import random
import time
from sklearn.metrics import f1_score, make_scorer
import statistics

import Helper.File.FileHelper as FileHelper
import PROPERTY
import Plot
import Preprocess.StatisticValue as PreStatVal
#import FeatureSelection.FeatureSelection as FeatureSelection
import dataToC.dataToC as dataToC
import dataFromC.dataFromC as dataFromC
import isolation_forest as myIsolationForest
import acquisition_function as acq_func

#%%
def MergeCsv():  
    #merge csv file (Span 1 ~ 6)
    for idx in range(0, len(PROPERTY.Experiment().MERGED_FILE_NAME)):
        FileHelper.MergeAllFile(PROPERTY.Experiment().PATH_RELATIVE[idx], '.csv', 
                                PROPERTY.Experiment().MERGED_FILE_NAME[idx], None, 
                                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                                PROPERTY.Experiment().FIRST_FILE, 
                                PROPERTY.Experiment().LAST_FILE)

def GetRawDataFromCsv():     
    # [0~5] for Each Span 1 ~ 6
    raw_data = []
    # Read CSV File
    for i in range(len(PROPERTY.Experiment().MERGED_FILE_NAME)):
        path = os.path.abspath(os.getcwd()) + PROPERTY.Experiment().PATH_RELATIVE[i] + '\\' + PROPERTY.Experiment().MERGED_FILE_NAME[i]
        merged_data = pd.read_csv(path, names=PROPERTY.Experiment().COL_NAME, index_col=False)
        raw_data.append(merged_data)
    return raw_data

def AddSpanLabel(X_processed):
    X_processed = copy.deepcopy(X_processed)
    span_processed_X = [] 
    
    # add label: span
    for i in range(len(X_processed)):
        span_processed_X.append(X_processed[i])
        span_processed_X[i]['Span'] = i+1
    return span_processed_X

def GetDataForFeatureSelection(X_processed):
    span_processed_X = AddSpanLabel(X_processed)   
    # list to df
    span_train_data = pd.DataFrame()
    for i in range(len(span_processed_X)):
        span_train_data = span_train_data.append(span_processed_X[i], ignore_index=True)
    return span_train_data

def ProduceFakeAbnormalData(abnormal_count, Xy_train):   
    # TODO
    fake_degree = 4
    fake_range = 1
    
    cols = Xy_train.columns.tolist()
    cols.remove('Span')
    fake_abnormal_data = pd.DataFrame(columns = cols)
    for col in cols:
        start = Xy_train[Xy_train.Span == 1].loc[:, col].mean()
        end = Xy_train[Xy_train.Span == 2].loc[:, col].mean()
        diff = end - start
        
        extremly_value = 0
        if diff > 0:
            extremly_value = start * fake_degree
            fake_abnormal_data.loc[0, col] = extremly_value
        else:
            extremly_value = 0
            fake_abnormal_data.loc[0, col] = 0
            
        for i in range(1, abnormal_count):
            if extremly_value == 0:
                fake_abnormal_data.loc[i, col] = random.uniform(extremly_value, extremly_value + fake_range)
            else:
                fake_abnormal_data.loc[i, col] = random.uniform(extremly_value, extremly_value - fake_range)
    return fake_abnormal_data
    


    


#%%
print('Expriment begin...')

'''
Raw data
'''
print('Raw data...')
# 如果有新的實驗檔案，要重新合併csv檔，把merge的註釋消掉
# MergeCsv()
raw_data = GetRawDataFromCsv()

print('Plot PLC cycle')
spans = ['span 1', 'span 2']
Plot.PlotMultiFeature(list_data = raw_data, feature = 'Current', 
                      xlim = PROPERTY.Experiment().PLC_CYCLE * 5, x_axis_label = 'Data Points', 
                      list_labels = spans, 
                      title = '5 PLC CYCLE')
Plot.PlotMultiFeature(list_data = raw_data, feature = 'Current', 
                      xlim = PROPERTY.Experiment().PLC_CYCLE, x_axis_label = 'Data Points', 
                      list_labels = spans, 
                      title = 'A PLC CYCLE')


#%%
'''
Preprocessing
'''
print('Preprocessing...')

X_processed = []

# 有新資料時，改成True
first_calculated = False
if first_calculated:
    # 把用不到的column拿掉，ex: 時間
    selected_col = ['Output_DC_Bus', 'Torque', 'Voltage',
                          'Power_Factor_Angle', 'RPM', 'Current']
    # 計算統計特徵值
    for i in range(0, len(raw_data)):
        df = pd.DataFrame()
        for j in range(0, len(selected_col)):
            df = pd.concat([df, PreStatVal.CalculateSixFeatureValue(raw_data[i], selected_col[j], PROPERTY.Experiment().FIRST_FILE, PROPERTY.Experiment().LAST_FILE)], axis=1)
        X_processed.append(df)
    # 刪掉null row    
    for i in range(0, len(X_processed)):
        X_processed[i] = X_processed[i].dropna()
    # save pickle
    for i in range(0, len(X_processed)):
        X_processed[i].to_pickle('pickle/processed_X_' + str(i))    
else:
    # read pickle
    for i in range(0, PROPERTY.Experiment().SPAN_COUNT):
        X_processed.append(pd.read_pickle('pickle/processed_X_' + str(i)))  

plot_processed_data = True
if plot_processed_data:
    print('Plot processed data')
    for i in range(X_processed[0].shape[1]):
        Plot.PlotMultiFeature(list_data = X_processed, feature = X_processed[0].columns[i], x_axis_label = 'Data Points', list_labels = spans, title = 'Feature of Different Span in ALL PLC Cycle')
        plt.figure(figsize=(7,5))


#%%
'''
Feature selection
'''
print('Feature selection...')

#
# train & test data for feature selection
#
all_data_label = GetDataForFeatureSelection(X_processed)

all_feat_data = all_data_label.drop('Span', axis=1)
label = all_data_label['Span']

fs_train_data, fs_test_data, fs_train_label, fs_test_label = train_test_split(
    all_feat_data, label, test_size=0.3, random_state=0,stratify=label)

display_multi_labels = ['1', '2']  
labels = [1, 2] 

print('\t 1. extra tree')
# extra_trees = FeatureSelection.ExtraTrees(150, fs_train_data, fs_train_label)
# extra_trees.Predict(fs_test_data)
# extra_trees.ConfusionMatrix(fs_test_label, display_multi_labels)
# extra_trees.PrintImportances()
# extra_trees.PlotImportances()

print('\t 2. random forest')
# random_forest = FeatureSelection.RandomForest( n_estimators = 150, max_features = "sqrt", 
#                                               max_depth = None, min_samples_leaf = 3, 
#                                               X = fs_train_data, y = fs_train_label)
# random_forest.Predict(fs_test_data)
# random_forest.ConfusionMatrix(fs_test_label, display_multi_labels)
# random_forest.PrintImportances()
# random_forest.PlotImportances()

print('\t 3. GBDT')
# gbdt = FeatureSelection.GradientBoostedDecisionTrees(n_estimators = 100, learning_rate = 1, 
#                                                      max_depth = 6, X = fs_train_data, y = fs_train_label)
# gbdt.Predict(fs_test_data)
# gbdt.ConfusionMatrix(fs_test_label, display_multi_labels)
# gbdt.PrintImportances()
# gbdt.PlotImportances()

import process_data
normal_span = [1]
abnormal_span = [2]
train_data, train_label = process_data.remove_partial_abnormal_data(all_data_label, 0.03, 
                                                                    normal_span, abnormal_span)

from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
IF = IsolationForest(contamination=0.03, random_state=0, max_samples=60, n_estimators=1000, max_features=36)
IF.fit(train_data)
score = IF.score_samples(fs_test_data) * -1
# Confusion Matrix
fs_test_pred_label = IF.predict(fs_test_data)
cm = confusion_matrix(fs_test_label, fs_test_pred_label)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 2])
# disp = disp.plot(include_values=True, cmap='Blues')
# plt.title('')


y_term = score - score.mean()
pearson_list = []
for i in range(fs_test_data.shape[1]):    
    x_term = fs_test_data.iloc[:, i] - fs_test_data.iloc[:, i].mean()
    # 分子
    numerator = (x_term * y_term).sum()
    # 分母
    denominator = ((x_term ** 2).sum() * (y_term ** 2).sum()) ** (1/2)
    pearson_list.append((numerator / denominator)**2)
    print('pearson ', i, ' = ', pearson_list[i])

print('---End---')

pearson = pd.Series(pearson_list, index=fs_test_data.columns)
pearson.sort_values(inplace=True)

from sklearn.feature_selection import f_regression
f_statistic, p_value = f_regression(fs_train_data, fs_train_label)

#seleted_features = ['Span', 'Power_Factor_Angle_Avg', 'Power_Factor_Angle_Max', 'Voltage_Avg', 'Voltage_Max', 'Current_Min']
seleted_features = ['Span', 'Power_Factor_Angle_Avg', 'Power_Factor_Angle_Max', 'Power_Factor_Angle_Min', 'Torque_Min', 'Torque_Avg']
# 測試pearson是否有用
seleted_features = ['Span', 'RPM_Max', 'Torque_Std', 'Current_Std', 'Power_Factor_Angle_Skewness', 'RPM_Skewness']
#seleted_features = ['Power_Factor_Angle_Avg','Current_Min',]

# data after feature selection
selected_data_label = all_data_label.loc[:, seleted_features]


#%%
'''
Train & Test data 
'''
print('Train & Test data...')
train_data_label, test_data_label, _, test_label = train_test_split(selected_data_label, label, test_size=0.3, random_state=0,stratify=label)
# 排序是為了後面的異常分數圖
test_data_label = test_data_label.sort_values(by=['Span'])
test_data = test_data_label.drop(columns = 'Span')
test_label = test_label.sort_values(ascending=True)

#
# 依異常比例，取部分異常資料
#
normal_span = [1]
abnormal_span = [2]
# = abnormal data 數量 / normal data 數量, range: 0~1
abnormal_ratio = (1.0 / 30.0)
import process_data
train_data, train_label = process_data.remove_partial_abnormal_data(train_data_label, abnormal_ratio, 
                                                                    normal_span, abnormal_span)


'''
C++
'''
print('data to C++...')
dataToC.DataToC(train_data, test_data_label)

# print('data from C++...')
# anomaly_score_from_C = dataFromC.DataFromC()
# for i in range(anomaly_score_from_C.shape[1]):
#     Plot.PlotFeature(anomaly_score_from_C, anomaly_score_from_C.columns[i])



'''
Isolation Forest
'''
print('Isolation Forest...')
IF = myIsolationForest.MyIsolationForest(abnormal_ratio, n_estimators=500, max_samples=20, max_features=5, X_train=train_data)

# label: 正常=1，異常=-1
test_bi_label = test_label.replace(to_replace = test_label[ test_label <= 1 ].tolist(), value=1 )
test_bi_label = test_bi_label.replace(to_replace = test_label[ test_label > 1 ].tolist(), value=-1 )

IF.Predict(test_data, test_bi_label, test_label, labels)
IF.PlotAnomalyScore('')

print('\t binary classification')
IF.ConfusionMatrixBinary([-1, 1], 'IF')
IF.ClassificationReportBinary()



#%%
print('GMM...')

from sklearn.mixture import GaussianMixture
anomaly_score = IF.all_score.values.reshape(-1, 1)
# 非常重要的兩個參數，會直接決定GMM的結果是否成功!
weights_init = [1/len(spans)] * len(spans)
means_init = np.array(IF.means).reshape(-1, 1)
# 因為是1-D的GMM，因此covariance_type='spherical'
gm = GaussianMixture(n_components=len(spans), covariance_type='spherical', 
                     random_state=0, init_params='kmeans', 
                     weights_init=weights_init, means_init=means_init).fit(anomaly_score)

# plot 1-D gaussian distribution
from scipy.stats import norm
plt.figure(figsize=(7, 5))

means = gm.means_
standard_deviations = gm.covariances_

for i in range(0, len(spans)):   
    x = np.arange(0.4, 0.8, 0.01)
    y = norm(means[i], standard_deviations[i])
    
    print('span', i+1)
    print('mean =', means[i], '\t std =', standard_deviations[i])
    
    plt.plot(x, y.pdf(x), label=spans[i])
    plt.legend()

prob = pd.DataFrame(gm.predict_proba(anomaly_score))

# result
true = IF.y_true_mul.reset_index(drop=True)
pred = IF.y_pred_mul.reset_index(drop=True)

true_pred_prob = pd.concat([true, pred], axis=1)
true_pred_prob.columns = ['True', 'Pred']
true_pred_prob = pd.concat([true_pred_prob, prob], axis=1)

true_pred_prob_wrong = true_pred_prob[true_pred_prob['True'] != true_pred_prob['Pred']]

true_pred_prob_wrong.rename(columns={0: 'prob_1', 1: "prob_2", 2: "prob_3", 3: "prob_4", 4: "prob_5", 5: "prob_6",}, inplace = True)

#%% 
import bayesian_optimizatoin
import bayesian_optimizatoin_plot as BOPlot

# 把警告訊息關掉
np.seterr(divide='ignore', invalid='ignore')
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


## 實驗: BO找到的最佳參數分布，是否收斂

# 實驗參數
exp_count = 10
bound = [ 
    {'name': 'abnormal_ratio',
      'type': 'fixed',
      'domain': abnormal_ratio}, 
        
    {'name': 'IF_count',
      'type': 'discrete',
      'domain': range(1, 2)}, # 設定range(想要的IF_count, IF_count+1)
    
    {'name': 'tree_count',
      'type': 'discrete',
      'domain': range(200, 201)},
    
    {'name': 'samples',
     'type': 'discrete',
     'domain': range(2, test_data.shape[0])},

    {'name': 'features',
     'type': 'discrete',
     'domain': range(3, 4)}
]
        
# run BO
opt_params = pd.DataFrame(columns=['score difference', 'tree_count', 'samples', 'features', 'time_cost'])
for i in range(exp_count):
    print('\n\t BO ex', i)
    time_start = time.time() #開始計時
    
    BO = bayesian_optimizatoin.BayesianOptimization(bound, train_data, test_data, test_bi_label)
    BO.init_samples(15, 'score_difference')
    opt_score_diff, opt_param = BO.run_native(25, 'EI')    
    
    time_end = time.time()    #結束計時
    time_cost= time_end - time_start   #執行所花時間
    print('\t', 'time cost', time_cost, 's')
    
    # 所有實驗的參數結果
    opt_params = opt_params.append({
        'score difference': opt_score_diff,
        'tree_count': opt_param[2], 
        'samples': opt_param[3], 
        'features': opt_param[4],
        'time_cost': time_cost}, ignore_index=True)
print(opt_params.describe())


# 存實驗結果
# opt_params.to_pickle('pickle/1000_BO_500TreeCount')

BOPlot.samples_distribution(opt_params['samples'])
BOPlot.features_distribution(opt_params)

#%%
m = np.array([[1, 1, 2], [1, 1, 3], [1, 1, 2]])
m = np.unique(m, axis=0)
print(m)
#%%
## 用BO找abnoraml ratio (暫停)

# 實驗參數
exp_count = 0
bound = [ 
    {'name': 'abnormal_ratio',
      'type': 'continous',
      'domain': range(0, 1)}, 
        
    {'name': 'IF_count',
      'type': 'discrete',
      'domain': range(30, 31)}, # 設定range(想要的IF_count, IF_count+1)
    
    {'name': 'tree_count',
      'type': 'discrete',
      'domain': range(200, 201)},
    
    {'name': 'samples',
     'type': 'discrete',
     'domain': range(60, 61)},

    {'name': 'features',
     'type': 'discrete',
     'domain': range(3, 4)}
]


opt_params = pd.DataFrame(columns=['abnormal_ratio', 'score_difference', 'tree_count', 'samples', 'features', 'time_cost'])
for i in range(exp_count):
    print('\n\t BO ex', i)
    time_start = time.time() #開始計時
    
    BO = bayesian_optimizatoin.BayesianOptimization(bound, train_data, test_data, test_bi_label)
    BO.init_samples(15, 'score_difference')
    opt_score_diff, opt_param = BO.run_abnormal_ratio(30, 'LCB', train_data_label)    
    
    time_end = time.time()    #結束計時
    time_cost= time_end - time_start   #執行所花時間
    print('\t', 'time cost', time_cost, 's')
    
    # 所有實驗的參數結果
    opt_params = opt_params.append({
        'abnormal_ratio': opt_param[0],
        'score_difference': opt_score_diff,
        'tree_count': opt_param[2], 
        'samples': opt_param[3], 
        'features': opt_param[4],
        'time_cost': time_cost}, ignore_index=True)
print(opt_params.describe())

#%%
## 實驗: 重複BO repeat次，對最佳參數取平均，看結果是否收斂
    
# 實驗參數
exp_count = 0
repeat = 3
bound = [ 
    {'name': 'abnormal_ratio',
      'type': 'fixed',
      'domain': abnormal_ratio}, 
        
    {'name': 'IF_count',
      'type': 'discrete',
      'domain': range(1, 2)}, # 設定range(想要的IF_count, IF_count+1)
    
    {'name': 'tree_count',
      'type': 'discrete',
      'domain': range(200, 201)},
    
    {'name': 'samples',
     'type': 'discrete',
     'domain': range(2, test_data.shape[0])},

    {'name': 'features',
     'type': 'discrete',
     'domain': range(3, 4)}
]
        
# run BO
opt_params = pd.DataFrame(columns=['score_difference', 'tree_count', 'samples', 'features', 'time_cost'])
for i in range(exp_count):
    print('\n\t BO ex', i)
    time_start = time.time() #開始計時
    temp_params = pd.DataFrame(columns=['score_difference', 'tree_count', 'samples', 'features'])
    
    # 每repeat次的最佳參數取平均
    for j in range(repeat):
        BO = bayesian_optimizatoin.BayesianOptimization(bound, train_data, test_data, test_bi_label)
        BO.init_samples(15, 'score_difference')
        temp_score_diff, temp_param = BO.run_native(25, 'LCB')  
        
        temp_params = temp_params.append({
            'score_difference': temp_score_diff,
            'tree_count': temp_param[2], 
            'samples': temp_param[3], 
            'features': temp_param[4]},ignore_index=True)
    
    time_end = time.time()    #結束計時
    time_cost= time_end - time_start   #執行所花時間
    print('\t', 'time cost', time_cost, 's')
    
    opt_param = temp_params.mean()    
    # 所有實驗的參數結果
    opt_params = opt_params.append({
        'score_difference': opt_param['score_difference'],
        'tree_count': opt_param['tree_count'], 
        'samples': opt_param['samples'], 
        'features': opt_param['features'],
        'time_cost': time_cost}, ignore_index=True)
print(opt_params.describe())

# 存實驗結果
# opt_params.to_pickle('pickle/repeat_')

BOPlot.samples_distribution(opt_params['samples'])
BOPlot.features_distribution(opt_params)
    
    
#%% 
## 實驗: 觀察相同的samples之下，建出來的IF的score difference分布

# 實驗參數
exp_count = 0
samples_try = [50, 100, 140]

# run ex
for j in range(len(samples_try)):
    score_differences = pd.Series([])
    if exp_count <= 0:
        break;
    # 重複建IF
    for i in range(exp_count):
        print(i)
        temp_if = myIsolationForest.MyIsolationForest(abnormal_ratio, n_estimators=200, max_samples=samples_try[j], max_features=3, X_train=train_data)
        temp_if.Predict(test_data, test_bi_label, test_label, labels)        
        # score difference取到小數點後3位
        score_differences[i] = round(temp_if.ScoreDifference(), 3)
        
    # 計算每種score difference出現的次數   
    sd_freq = score_differences.value_counts()

    #
    # 畫score differnce 分布
    #
    plt.figure(figsize=(20,5))
    plt.grid(True)
     
    plt.plot(sd_freq.index, sd_freq, 'o')
    
    # 標每個點的y值
    for a, b in zip(sd_freq.index, sd_freq):
        plt.text(a, b, str(b), ha='left', va='top')
    
    plt.xlabel('score difference')
    plt.ylabel('frequency')
    plt.title('samplels =' + str(samples_try[j]))
    
    
#%%
## 實驗: stable tree count 

import tree_count_search

# 實驗參數
# 初始設一個極大的tree count(保證穩定)，然後用BO找到feature, samples
stable_tree_count_list = [200]
# 設定穩定度，會找到此穩定度對應的tree count
stable_threshold = 0.5

samples_list = []
features_list = []
stable_score_diff_list = []
all_tree_count = []

# 迭代
iteration_count = 0
for i in range(iteration_count):
    print('Iteration', i)
    #
    # ----- Step 1: 固定tree count，用BO找samples、feature -----
    #
    bound = [ 
        {'name': 'abnormal_ratio',
          'type': 'fixed',
          'domain': abnormal_ratio}, 
            
        {'name': 'IF_count',
          'type': 'discrete',
          'domain': range(1, 2)}, # 設定range(想要的IF_count, IF_count+1)
        
        {'name': 'tree_count',
          'type': 'discrete',
          'domain': range(stable_tree_count_list[i], stable_tree_count_list[i]+1)},
        
        {'name': 'samples',
         'type': 'discrete',
         'domain': range(2, test_data.shape[0]-100)},
    
        {'name': 'features',
         'type': 'discrete',
         'domain': range(3, 6)}
    ]
    
    print('Step 1')
    print('stable_tree_count = ', stable_tree_count_list[i])
    
    BO = bayesian_optimizatoin.BayesianOptimization(bound, train_data, test_data, test_bi_label)
    BO.init_samples(15, 'score_difference')
    opt_score_diff, opt_param = BO.run_native(25, 'LCB')   

    # 第一階段找到的最佳samples, features
    # 代入第二階段
    print('opt_samples =', opt_param[3])
    print('opt_features =', opt_param[4])
    samples_list.append(int(opt_param[3]))
    features_list.append(int(opt_param[4]))   


    #
    # ----- Step 2: 固定samples、features，bineary search找tree count -----
    #
    print('Step 2')
    
    score_diff_list, all_tree_count, stable_tree_count, stable_score_diff = tree_count_search.binary_search_dynamic_high(
        low = 2, high = 200, 
        stop_interval = 50, threshold=stable_threshold,
        abnormal_ratio = bound[0]['domain'],
        samples = opt_param[3], 
        features = opt_param[4],
        stable_tree_count_func = unknown_function.stable_tree_count,
        train_data = train_data, test_data = test_data, test_bi_label = test_bi_label)
    
    print('minimum stable tree count =', stable_tree_count)
    stable_tree_count_list.append(stable_tree_count)
    print('stable_score_diff =', stable_score_diff)
    stable_score_diff_list.append(stable_score_diff)
        
# 刪除初始設定的很大的 tree count，方便後面作圖
del(stable_tree_count_list[0])
#%%
if iteration_count > 0:
    # 畫binary search的過程
    plt.figure(figsize=(20,17))
    plt.grid(True)
    plt.plot(all_tree_count, score_diff_list, 'o:')
    
    # 畫 stable threhold
    plt.plot([0, 100], [stable_threshold, stable_threshold], '-')
    
    for a, b in zip(all_tree_count, score_diff_list):
        plt.text(a, b, str(a), ha='left', va='top')
    
    plt.xlabel('tree count')
    plt.ylabel('score difference  (%)')
#%% 
# 畫BO找到的數值，隨著iteration的變化

fig_width = 20

#
# stable tree count
#
plt.figure(figsize=(fig_width,5))
plt.grid(True)

plt.plot(range(iteration_count), stable_tree_count_list, 'o:')
plt.xlabel('iteration', fontsize=30)
plt.ylabel('tree count', fontsize=30)


#
# stable_score_diff
#
plt.figure(figsize=(fig_width,5))
plt.grid(True)
plt.plot(range(iteration_count), stable_score_diff_list, 'o:')
plt.xlabel('iteration', fontsize=30)
plt.ylabel('stable percentage', fontsize=30)

#
# smaples
#
plt.figure(figsize=(fig_width,5))
plt.grid(True)
plt.plot(range(iteration_count), samples_list, 'o:')
plt.xlabel('iteration', fontsize=30)
plt.ylabel('samples', fontsize=30)

#
# features
#
plt.figure(figsize=(fig_width,5))
plt.grid(True)
plt.plot(range(iteration_count), features_list, 'o:')
plt.xlabel('iteration', fontsize=30)
plt.ylabel('features', fontsize=30)   

#%%
'''
Bayesian Optimization Isolation Forest 
'''
print('Bayesian Optimization Forest...')
# BO_IF = myIsolationForest.MyIsolationForest(abnormal_ratio, n_estimators=50, max_samples=100, max_features=5, X_train=train_data)
BO_IF = myIsolationForest.MyIsolationForest(abnormal_ratio, n_estimators=200, max_samples=60, 
                             max_features=3, X_train=train_data, random_state=None)
BO_IF.Predict(test_data, test_bi_label, test_label, labels)
BO_IF.PlotAnomalyScore('BO')

print('\t binary classification')
# label: 正常=1，異常=-1
BO_IF.ConfusionMatrixBinary([1, 2], 'BO IF')
BO_IF.ClassificationReportBinary()

BO_IF.PlotScoreHist(10, 2)


print('---End---')
