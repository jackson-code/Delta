# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 20:22:58 2021

@author: user
"""

import pandas as pd


def CalculateSixFeatureValue(df, col_name, first_file, last_file):
    col = [col_name + '_Avg', col_name + '_Min', col_name + '_Max',
                   col_name + '_Std', col_name + '_Skewness', col_name + '_Kurtosis']
    result = pd.DataFrame(columns=col)

    for i in range(first_file, last_file):
        _list = []
        _list.append(df[col_name][df.PLC_cycle == i].mean())
        _list.append(df[col_name][df.PLC_cycle == i].min())
        _list.append(df[col_name][df.PLC_cycle == i].max())
        _list.append(df[col_name][df.PLC_cycle == i].std())
        _list.append(df[col_name][df.PLC_cycle == i].skew())
        _list.append(df[col_name][df.PLC_cycle == i].kurt())

        temp = pd.DataFrame([_list], columns=col)
        result = result.append(temp, ignore_index=True)

    return result

