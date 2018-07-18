# coding:utf-8
import time
import pandas as pd
from utils import *


def read_in_process_feature(index_1, index_2):
    # 读取数据
    part_1 = pd.read_csv(index_1, sep='$', dtype=str)
    part_2 = pd.read_csv(index_2, sep='$', dtype=str)
    part_1_2 = pd.concat([part_1, part_2])
    part_1_2 = pd.DataFrame(part_1_2).sort_values('vid').reset_index(drop=True)
    begin_time = time.time()
    print('begin')
    bad_word_list_single = ['-', '无', '未查', '弃查', '未要求检查', '指数', '手动', '降脂后复查',
                            '因无法配合不能检查', '未见', '标本已退检', '\/', '未做', '>', '<',
                            '见TCT', '见刮片', '次/分', '未检']
    bad_word_list_mode = ['详见', '分析报告', '已退检', '未做', '---', '\d+月\d+日']
    part_1_2.dropna(axis=0, subset=['field_results'], inplace=True)
    part_1_2 = part_1_2[~part_1_2['field_results'].isin(bad_word_list_single)]
    for mode in bad_word_list_mode:
        part_1_2 = part_1_2[~part_1_2['field_results'].str.contains(mode)]
    part_1_2['field_results'] = part_1_2['field_results'].apply(strQ2B)
    part_1_2['field_results'] = pd.to_numeric(part_1_2['field_results'], errors='ignore')
    part_1_2 = part_1_2.drop_duplicates(keep='first')

    # 重复数据的拼接操作
    def merge_table(df):
        try:
            df['field_results'] = pd.to_numeric(df['field_results'])
            all_num = list(df['field_results'])
            ave = 0
            for a_num in all_num:
                ave += a_num
            ave /= len(all_num)
            return ave
        except ValueError:
            df['field_results'] = df['field_results'].astype(str)
            if df.shape[0] > 1:
                merge_df = ",".join(list(df['field_results']))
            else:
                merge_df = df['field_results'].values[0]
            return merge_df

    # 数据简单处理
    print('find_is_copy')
    print(part_1_2.shape)
    is_happen = part_1_2.groupby(['vid', 'table_id']).size().reset_index()
    # 重塑index用来去重
    is_happen['new_index'] = is_happen['vid'] + '_' + is_happen['table_id']
    is_happen_new = is_happen[is_happen[0] > 1]['new_index']

    part_1_2['new_index'] = part_1_2['vid'] + '_' + part_1_2['table_id']

    unique_part = part_1_2[part_1_2['new_index'].isin(list(is_happen_new))]
    unique_part = unique_part.sort_values(['vid', 'table_id'])
    no_unique_part = part_1_2[~part_1_2['new_index'].isin(list(is_happen_new))]
    print('begin')
    part_1_2_not_unique = unique_part.groupby(['vid', 'table_id']).apply(merge_table).reset_index()
    part_1_2_not_unique.rename(columns={0: 'field_results'}, inplace=True)
    print('xxx')
    tmp = pd.concat([part_1_2_not_unique, no_unique_part[['vid', 'table_id', 'field_results']]])
    # 行列转换
    print('finish')
    tmp = tmp.pivot(index='vid', values='field_results', columns='table_id')
    print(tmp.shape)
    print('total time', time.time() - begin_time)
    tmp.to_csv('../data/tmp_feature_raw.csv')


def read_in_process_label(index):
    train = pd.read_csv(index, encoding='gb18030')

    # 数据清洗
    def data_clean(df):
        for c in ['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白']:
            df[c] = df[c].apply(clean_label)
            df[c] = df[c].astype('float64')
        return df

    train = data_clean(train)
    for select in ['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白']:
        feat = ['vid', select]
        train_1 = train[feat]
        train_1[train_1[select] <= 0] = None
        train_1[train_1[select] > 500] = None
        a = train_1.dropna(axis=0, how='any', thresh=None)
        a.to_csv('../data/label_' + select + '.csv', encoding='gb18030', index=False)


if __name__ == '__main__':
    data_index_1 = '../data/meinian_round1_data_part1_20180408.txt'
    data_index_2 = '../data/meinian_round1_data_part2_20180408.txt'
    label_index = '../data/meinian_round1_train_20180408.csv'
    read_in_process_feature(data_index_1, data_index_2)
    read_in_process_label(label_index)
