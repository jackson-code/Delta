# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 20:35:15 2021

@author: user
"""
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import pandas as pd

class IF():
    def __init__(self, abnormal_ratio, n_estimators, max_samples, max_features, X_train, random_state=None):
        self.IF = IsolationForest(
            contamination=abnormal_ratio, random_state=random_state, max_samples=max_samples, n_estimators=n_estimators, max_features=max_features)
        self.IF.fit(X_train)
        
    def PlotScoreHist(self, normal_bins, anomaly_bins):
        plt.figure(figsize=(7,5))
        plt.title('')
        
        # 畫正常資料的長條圖
        nums,bins,patches = plt.hist(self.normal_score, bins=normal_bins,  
                                     density=False, alpha=0.3, rwidth=0.5,
                                     label='normal testing dataset')
        #plt.xticks(bins,bins) 
        for num,bin in zip(nums,bins):
            plt.annotate(num,xy=(bin,num))
        
        # 畫異常資料的長條圖
        nums,bins,patches = plt.hist(self.abnormal_score, bins=anomaly_bins, 
                                     density=False, alpha=0.3, rwidth=0.5,
                                     label='abnormal testing dataset')
        #plt.xticks(bins,bins) 
        for num,bin in zip(nums,bins):
            plt.annotate(num,xy=(bin,num))
        
        plt.legend(loc='upper left')
        plt.xlabel("anomaly score")
        plt.ylabel("data quantity ")
        
    def PlotAnomalyScore(self, title):    
        axis_x = np.arange(0, len(self.all_score))
    
        plt.figure(figsize=(7,5))
    
        plt.title(title, fontsize=15, y=1.05, fontweight='bold')
        plt.ylabel('Anomaly Score', fontsize=15) 
        plt.xlabel('Data Points', fontsize=15) 
        plt.grid()
    
        plt.plot(axis_x, self.all_score)
        
    def Predict(self, X_test, y_test_bi, y_true_mul, labels):        
        self.X_test = X_test
        
        # 先資料和label合併，才能將正常、異常資料分開
        Xy_test = pd.concat([X_test, y_test_bi], axis=1)
        
        self.X_test_bi_normal = Xy_test[Xy_test['Span'] == 1]
        self.X_test_bi_normal.drop(labels='Span', axis=1, inplace=True)
        
        self.X_test_bi_abnormal = Xy_test[Xy_test['Span'] == -1]
        self.X_test_bi_abnormal.drop(labels='Span', axis=1, inplace=True)
                
        self.y_test_bi = y_test_bi
        
        # 將正常、異常分開預測，為了畫圖時兩者顏色分開
        self.y_test_bi_normal = self.IF.predict(self.X_test_bi_normal)
        self.y_test_bi_abnormal = self.IF.predict(self.X_test_bi_abnormal)
        
        # binary classification (原本IF的結果)
        self.y_pred_bi = np.r_[self.y_test_bi_normal, self.y_test_bi_abnormal]
        
        # 正常資料的anomaly score
        score = self.IF.score_samples(self.X_test_bi_normal)
        self.normal_score = pd.Series(score * -1, index=self.X_test_bi_normal.index, name='Anomaly_score')
        
        # 異常資料的anomaly score
        score = self.IF.score_samples(self.X_test_bi_abnormal)
        self.abnormal_score = pd.Series(score * -1, index=self.X_test_bi_abnormal.index, name='Anomaly_score')
        
        self.all_score = pd.concat([self.normal_score, self.abnormal_score], axis=0)
        
        # multiply classification
        self.y_true_mul = y_true_mul
        self.y_pred_mul = self.MultiClassify(y_true_mul, labels)

        
    def MultiClassify(self, y_true, labels):
        df = pd.concat([y_true, self.all_score], axis=1, ignore_index=True)
        df.columns = ['Span', "Anomaly_score"]
        
        # 計算每種span的異常分數的平均
        mean = []
        for i in labels:
            mean.append(df[df.Span == i].Anomaly_score.mean())
        self.means = mean
        
        # 計算平均之間的中點，當作multi-classify的threshold   
        mean_middle = []
        for i in range(len(labels)):
            # 避免超出idx範圍
            if i == len(labels)-1:
                break
            current = mean[i]
            next = mean[i+1]
            mean_middle.append((current + next) / 2)
        self.multi_threshold = mean_middle
        

        score = pd.Series.to_list(self.all_score)
        y_pred = []
        # 多元分類
        # 將每個異常分數與threshold合併並排序，異常分數的idx+1即為prediction
        for i in range(len(score)):
            temp = [score[i]]
            temp.extend(mean_middle)
            temp.sort()
            y_pred.append(temp.index(score[i]) + 1)
        y_pred = pd.Series(y_pred, index=y_true.index)
        return y_pred
    
    def ConfusionMatrixBinary(self, display_labels, title):   
        # binary classification (原本IF的結果)      
        cm = confusion_matrix(self.y_test_bi, self.y_pred_bi)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp = disp.plot(include_values=True, cmap='Blues')
        plt.title(title)
        
        
    def ClassificationReportBinary(self):
        print(classification_report(self.y_test_bi, self.y_pred_bi)) 
        
    def ConfusionMatrixMulti(self, display_labels, title):   
        # multi classification 
        plt.title(title)
        cm = confusion_matrix(self.y_true_mul, self.y_pred_mul)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp = disp.plot(include_values=True, cmap='Blues')             
        
    def ClassificationReportMulti(self):
        print(classification_report(self.y_true_mul, self.y_pred_mul)) 
        
    def ConfusionMatrixThree(self, display_labels, title):   
        # 把span1, 2合併，3 4 合併，5 6 合併
        y_pred_3 = self.y_pred_mul.copy()
        y_pred_3[self.y_pred_mul==1] = 12
        y_pred_3[self.y_pred_mul==2] = 12
        y_pred_3[self.y_pred_mul==3] = 34
        y_pred_3[self.y_pred_mul==4] = 34
        y_pred_3[self.y_pred_mul==5] = 56
        y_pred_3[self.y_pred_mul==6] = 56
        self.y_pred_3 = y_pred_3
        y_true_3 = self.y_true_mul.copy()
        y_true_3[self.y_true_mul==1] = 12
        y_true_3[self.y_true_mul==2] = 12
        y_true_3[self.y_true_mul==3] = 34
        y_true_3[self.y_true_mul==4] = 34
        y_true_3[self.y_true_mul==5] = 56
        y_true_3[self.y_true_mul==6] = 56
        self.y_true_3 = y_true_3
        
        plt.title(title)
        cm = confusion_matrix(self.y_true_3, self.y_pred_3)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp = disp.plot(include_values=True, cmap='Blues')             
        
    def ClassificationReportThree(self):
        print(classification_report(self.y_true_3, self.y_pred_3)) 
        
    
        