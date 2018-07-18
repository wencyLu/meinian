# coding:utf-8
import pandas as pd
import numpy as np
import lightgbm as lgbm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import time
import datetime
from utils import *

cls_dict = {'收缩压': 0.02, '舒张压': 0.02, '血清甘油三酯': 0.03, '血清高密度脂蛋白': 0.04, '血清低密度脂蛋白': 0.04}


def truncate(df): #去除很大或者很小值
    for i in df.columns[1:]:
        temp_df = df[i][df[i].notnull()].copy()
        llt = np.percentile(temp_df.values, 0.05)  #小于0.05%
        ult = np.percentile(temp_df.values, 99.95) #大于99.95%
        temp_df[temp_df > ult] = ult
        temp_df[temp_df <= llt] = llt
        df[i] = temp_df
    return df


def evalerror(pred, df):
    label = df.get_label().values.copy()
    score = mean_squared_error(label, pred)
    return 'mse', score, False


def loge(df):
    df = np.log(df + np.ones(len(df)))
    return df


def powerr(df):
    df = np.exp(df) - np.ones(len(df))
    return df


def train(strs, count, feature=None):
    params = {
        'learning_rate': cls_dict[strs],
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mse',             # 使用均方误差
        'num_leaves': 60,            # 最大叶子数for base learner
        'feature_fraction': 0.6,     # 选择部分的特征
        'min_data': 100,             # 一个叶子上的最少样本数
        'min_hessian': 1,            # 一个叶子上的最小 hessian 和，子叶权值需大于的最小和
        'verbose': 1,
        'lambda_l1': 0.3,            # L1正则项系数
        'device': 'cpu',
        'num_threads': 8,            # 最好设置为真实核心数
    }
    path = '../data/'
    filename = 'tmp_feature_final.csv'
    train_df = pd.read_csv(path+filename, encoding='gb18030')
    train_df = train_df.drop(['04341', '04342', '04343', '04344', '04345','0101_0','0101_1','0101_2','0101_3','0101_4','0102_0','0102_1','0102_2','0102_3','0102_4'], axis=1)
    filename = 'meinian_round1_test_b_20180505.csv'
    test_df = pd.read_csv(path+filename, encoding='gb18030')
    test_df = test_df.merge(train_df, how='inner', on='vid')
    filename = 'meinian_round1_train_20180408.csv'
    train_df2 = pd.read_csv(path + filename, encoding='gb18030')
    s = pd.DataFrame(train_df2['vid'].copy())
    train_df = train_df.merge(s, how='inner', on='vid')
    filename1 = '../data/label_' + strs + '.csv'
    with open(filename1, encoding='gb18030') as f:
        label_df = pd.read_csv(f)
    train_df = truncate(train_df)

    submission = np.zeros(test_df.shape[0])

    if feature is None:
        feature = train_df.columns[1:]
        #'0102_0','0102_1','0102_2','0102_3','0102_4'
        print(feature.shape)

    label_df.iloc[:, 1] = loge(label_df.iloc[:, 1])
    train_df = train_df.merge(label_df, how='inner', on='vid')
    kfo = KFold(5, shuffle=True)
    t0 = time.time()
    loss = 0
    for j in range(5):
        train_index, valid_index = next(kfo.split(train_df))
        print('训练{}ing...第{}次'.format(strs, j+1))
        train_data = train_df.iloc[train_index]
        vali_data = train_df.iloc[valid_index]
        train_data_set = lgbm.Dataset(train_data[feature], train_data[strs])
        vali_data_set = lgbm.Dataset(vali_data[feature], vali_data[strs])
        model = lgbm.train(params, train_data_set, num_boost_round=3000, valid_sets=vali_data_set, verbose_eval=100,
                           feval=evalerror, early_stopping_rounds=100)
        print('CV训练已经用时{}秒'.format(time.time() - t0))
        train_pred = model.predict(vali_data[feature])
        loss += mean_squared_error(vali_data[strs], train_pred)
        submission += model.predict(test_df[feature])
        feat_imp = pd.Series(model.feature_importance(), index=feature).sort_values(ascending=False)
    return feat_imp, loss/5, submission/5


def select_feat(f_imp0, drop_rate):
    size = f_imp0.shape[0]
    f_imp_lst = list(f_imp0.index.values)
    f1 = f_imp_lst[:int(size - drop_rate * size)]
    return f1

def train_head():
    target_lst = ['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白']
    filename = '../data/meinian_round1_test_b_20180505.csv'
    test_df = pd.read_csv(filename, encoding='gb18030')
    feature_imp = []
    count = 0
    loss = 0
    a=[]
    for i in target_lst:
        f_imp, loss_single, submission = train(i,count)
        feature_imp.append(f_imp)
        loss += loss_single
        a.append(loss_single)
        submission = powerr(submission)
        test_df[i] = submission
        count += 1
    feature=pd.DataFrame(feature_imp)
    print('5折线下CV损失', loss/5, a)
    return a
