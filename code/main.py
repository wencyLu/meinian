# coding:utf-8
import sys
import os
sys.path.append('../code')
import pandas as pd
from read_data_initial import read_in_process_label, read_in_process_feature
from preprocess_data import select_process
from operation_list_condense import feature_process
from trainmodel import *
pd.options.mode.chained_assignment = None


def main():
    
    print('数据拼接和标签处理')
    data_index_1 = '../data/meinian_round1_data_part1_20180408.txt'
    data_index_2 = '../data/meinian_round1_data_part2_20180408.txt'
    label_index = '../data/meinian_round1_train_20180408.csv'

    # Path data will save at
    temp_name_1 = '../data/tmp_feature_raw.csv'
    temp_name_2 = '../data/tmp_feature_select.csv'
    temp_name_3 = '../data/tmp_feature_final.csv'

    read_in_process_label(label_index)
    if os.path.exists(temp_name_1):
        pass
    else:
        read_in_process_feature(data_index_1, data_index_2)

    print('数据预处理')
    if os.path.exists(temp_name_2):
        pass
    else:
        select_process()
    if os.path.exists(temp_name_3):
        pass
    else:
        feature_process()
    
    print('开始训练')

    a1 = train_head()
    a2 = train_head()
    a3 = train_head()
    a4 = train_head()
    a5 = train_head()
    a6 = train_head()
    a7 = train_head()
    a8 = train_head()
    print(a1,a2,a3,a4,a5,a6,a7,a8)


if __name__=='__main__':
    main()
