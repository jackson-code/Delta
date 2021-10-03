# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 19:42:57 2021

@author: user
"""

import matplotlib.pyplot as plt
import numpy as np

COLOR = ['blue', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'pink']

def PlotFeature(data, feature, y_title='', title='', save=False):
    
    plt.figure(figsize=(7,5))
    
    plt.title(title, fontsize=15, y=1.05, fontweight='bold')

    # x axis
    x = np.arange(0, len(data))
    plt.xlabel('Data Points', fontsize=15)       

    # y axis
    y = data[feature]
    if y_title == '':
        plt.ylabel(feature, fontsize=15)        
    else:
        plt.ylabel(y_title, fontsize=15)        
    
    plt.grid()
    plt.plot(x, y, color='blue')
    
    if save==True:
        plt.savefig(fname='picture/' + feature)  
        
        
        
def PlotMultiFeature(list_data, feature, x_axis_label, list_labels, title, xlim=0):
    # check params
    if len(list_data) != len(list_labels):
        print('ERROR: length of data doesnt equal to length of labels')
    
    plt.figure(figsize=(7, 5))

    title = title
    plt.title(title, fontsize=15, y=1.05, fontweight='bold')
    
    plt.xlabel(x_axis_label, fontsize=15)
    plt.ylabel(feature, fontsize=15)
    
    plt.grid()
    
    # Range of Axes x
    if xlim != 0:
        plt.xlim(0, xlim)
    
    for i in range(0, len(list_data)):
        x = np.arange(0, len(list_data[i][feature]))
        y = list_data[i][feature]
        
        plt.plot(x, y, color = COLOR[i], label=list_labels[i])
        plt.legend()