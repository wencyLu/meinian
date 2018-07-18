# coding:utf-8
import pandas as pd
import numpy as np
import re


def strQ2B(ss):
    ustring = str(ss)
    if ustring == 'nan':
        return ss
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring


def out_other(ss):
    in_str = str(ss)
    if in_str == 'nan':
        return ss
    temp = re.findall(r'\d+\.\d+', in_str)
    if len(temp) == 0:
        return ss
    else:
        return temp[0]


def nums_ave(ss):
    in_str = str(ss)
    chamber = re.findall(r'\d+(?:\.\d+)?', in_str)
    if len(chamber) == 0:
        return ss
    else:
        ave = 0
        for nums in chamber:
            ave += float(nums)
        ave /= len(chamber)
        return ave


def dot_replace(ss):
    in_str = str(ss)
    dot_list = re.findall('。', in_str)
    if len(dot_list) == 0:
        return ss
    else:
        out_str = re.sub('。', '.', in_str)
        return out_str


def two_dot_replace(ss):
    in_str = str(ss)
    dot_list = re.findall('\.\.', in_str)
    if len(dot_list) == 0:
        return ss
    else:
        out_str = re.sub('\.\.', '.', in_str)
        return out_str


def my_concat(total_list, add_on_series):
    list_total = list(total_list.columns)
    if type(add_on_series) == list:
        for series in add_on_series:
            total_list = my_concat(total_list, series)
            # if series.name in list_total:
            #     print(series.name + ' has been added!!')
            # else:
            #     total_list = pd.concat([total_list, series], axis=1)
    elif type(add_on_series) == pd.DataFrame:
        total_list = pd.concat([total_list, add_on_series], axis=1)
    else:
        series = add_on_series
        if series.name in list_total:
            print(series.name + ' has been added!!')
        else:
            total_list = pd.concat([total_list, series], axis=1)
    return total_list


def clean_label(x):
    x = str(x)
    if '+' in x:  # 16.04++
        i = x.index('+')
        x = x[0:i]
    if '>' in x:  # > 11.00
        i = x.index('>')
        x = x[i+1:]
    if len(x.split(sep='.')) > 2:  # 2.2.8
        i = x.rindex('.')
        x = x[0:i] + x[i+1:]
    if '未做' in x or '未查' in x or '弃查' in x or '-' in x:
        x = np.nan
    if (not str(x).isdigit()) and len(str(x)) > 4:
        x = x[0:4]
    return x
