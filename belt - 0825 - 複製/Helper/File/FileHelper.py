# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 21:21:38 2021

@author: user
"""


import pandas as pd
import os

def MergeAllFile(relative_path, file_name_extenstion, new_file_name, cols_new_name, cols_merged, start_file=0, end_file=None):
    '''
    Description:
        Merge multiple file to a single file.
        Merging order is by sorted file name. 
    Note:
        1. File name must be number, and the function will sorting the file name before merging.
    Argument:
        1. relative_path
            type: ''
            path relative to the python code, ex: '\\your_folder'
        2. file_name_extenstion
            type: ''
            type of files, ex: '.txt' 
        3. new_file_name
            type: ''
        4. cols_new_name
            type: None or ['']
            None: don't add cols name in merged file
            ['string']: a list of cols name you want to add in merged file
        5. cols_merged
            type: [int]
            specify what columns you want to merged, ex: [0, 3, 4]
    Problem:
        Row one can't be merged.
    '''
    
    #
    # Get all file name and sort
    #
    path = os.path.abspath(os.getcwd()) + relative_path  # 文件夾路徑
    short_all_name = []
    for file_name in os.listdir(path):
        # 確定檔案類型
        if os.path.splitext(file_name)[1] == file_name_extenstion \
                and file_name != new_file_name:
            t = int(os.path.splitext(file_name)[0])  # 取得檔名

            if start_file == 0:
                if end_file == None or t <= end_file:
                    short_all_name.append(t)
            elif t >= start_file:
                if end_file == None or t <= end_file:
                    short_all_name.append(t)

    short_all_name.sort()

    #
    # Add file type in all file
    #
    file_all_name = []  # 用來儲存所有文件的名字
    for file_name in short_all_name:
        file_all_name.append(str(file_name) + file_name_extenstion)

    #
    # Add cols name or not
    #
    if cols_new_name == None:
        df = pd.DataFrame().T
    else:
        # Add cols_new_name in first row in FileHelper file
        df = pd.DataFrame(cols_new_name).T

    #
    # Merge file
    #
    try:
        print('\nStart merge...')
        df.to_csv(path + '/' + new_file_name,
                  encoding='gbk', header=False, index=False)
        for fn in file_all_name:
            data = pd.read_csv(path + '/' + fn)
            print(fn, end=' ')
            #data = data.iloc[0:, cols_merged]  # 跳過標題行
            data.to_csv(path + '/' + new_file_name, mode='a',
                        encoding='gbk', header=True, index=False)
        print('\nFinish, new file: ' + new_file_name)
    except PermissionError as e:
        print('Error:' + str(type(e)) + '！\nIs file opening?\n Please close.')


