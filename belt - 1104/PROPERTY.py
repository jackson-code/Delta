# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 20:25:53 2021

@author: user
"""

# 如果有新的實驗檔案，需修改
# span 1 張力最大，span 5,6 張力最小(視為異常) 
class Experiment():
    def __init__(self):
        self        
    #
    # File
    #
    SPAN_COUNT = 2
    MERGED_FILE_NAME = ['merged_3.csv', 'merged_6.csv',]    
    PATH_RELATIVE = ['\\data-0825\\3', '\\data-0825\\6',]    
    COL_NAME = ['PLC_cycle', 'Time_Stamp', 'Torque_cmd', 'Frequency', 'Output_DC_Bus',
        'Power_Factor_Angle', 'Torque', 'RPM', 'Voltage', 'Current']    
    # 每個span的前20個csv不採用，因為PLC正在切換span，導致訊號不穩定
    FIRST_FILE = 20
    LAST_FILE = 500
    
    
    #      
    # PLC
    #
    PLC_CYCLE = 120   
    

        
