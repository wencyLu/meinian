import pandas as pd
import numpy as np
import re
import os
import jieba
import jieba.posseg as psg
import gensim
from gensim.models import word2vec, Doc2Vec
from collections import Counter
import codecs
from utils import *
pd.options.mode.chained_assignment = None

def clean_11__(feature):
    index_format = '11%02d'
    in_list = [6, 7, 10, 12, 15, 17, 27]
    hold_play = []
    for index in in_list:
        index_name = index_format % index
        try:
            print('start ' + index_name)
            wait_play = feature[index_name]
            wait_play = wait_play.apply(strQ2B)
            wait_play = wait_play.apply(nums_ave)

            try:
                wrong_index = []
                for mode in wrong_index:
                    wait_play[wait_play.str.contains(mode, na=False)] = np.nan

            except AttributeError:
                pass

            try:
                wait_play = pd.to_numeric(wait_play)
            except ValueError:
                print('没处理好数据啊！！！！！')
                raise ValueError
            print('finished')

            hold_play.append(wait_play)

        except KeyError:
            print(index_name + ' no exist!')

    return hold_play
def clean_1000_(feature):
    index_format = '1000%d'
    in_list = [2, 3, 4, 9]
    hold_play = []
    for index in in_list:
        index_name = index_format % index
        try:
            print('start ' + index_name)
            wait_play = feature[index_name]
            wait_play = wait_play.apply(strQ2B)
            # wait_play = wait_play.apply(two_dot_replace)

            try:
                wrong_index = []
                for mode in wrong_index:
                    wait_play[wait_play.str.contains(mode, na=False)] = np.nan
            except AttributeError:
                pass

            wait_play = wait_play.apply(nums_ave)

            try:
                wait_play = pd.to_numeric(wait_play)
            except ValueError:
                print('没处理好数据啊！！！！！')
                raise ValueError
            print('finished')

            hold_play.append(wait_play)

        except KeyError:
            print(index_name + ' no exist!')

    return hold_play
def clean_18__(feature):
    index_format = '18%02d'
    in_list = [14, 15, 40, 45, 50, 73]
    hold_play = []
    for index in in_list:
        index_name = index_format % index
        try:
            print('start ' + index_name)
            wait_play = feature[index_name]
            wait_play = wait_play.apply(strQ2B)
            wait_play = wait_play.apply(dot_replace)
            wait_play = wait_play.apply(nums_ave)

            try:
                wrong_index = ['CLT1D', '阴性', '\+']
                for mode in wrong_index:
                    wait_play[wait_play.str.contains(mode, na=False)] = np.nan

            except AttributeError:
                pass

            try:
                wait_play = pd.to_numeric(wait_play)
            except ValueError:
                print('没处理好数据啊！！！！！')
                raise ValueError
            print('finished')

            hold_play.append(wait_play)

        except KeyError:
            print(index_name + ' no exist!')

    return hold_play
def clean_2302(feature):
    print('start 2302')
    wait_play = feature['2302']

    # 症状表
    symptom = [['亚健康'],
               ['健康'],
               ['疾病']]
    for cls, mode_list in enumerate(symptom):
        for mode in mode_list:
            wait_play[wait_play.str.contains(mode, na=False)] = '2302_'+str(cls)
    wait_play = wait_play.fillna('2302_' + str(len(symptom)))

    wait_play = pd.get_dummies(wait_play)
    wait_play = wait_play.drop(columns='2302_' + str(len(symptom)))
    print('finished')
    return wait_play
def clean_23__(feature):
    index_format = '23%02d'
    in_list = [33, 71, 72, 76, 86, 90]
    hold_play = []
    for index in in_list:
        index_name = index_format % index
        try:
            print('start ' + index_name)
            wait_play = feature[index_name]
            wait_play = wait_play.apply(strQ2B)
            wait_play = wait_play.apply(two_dot_replace)

            try:
                wrong_index = ['2.792.20', '阴性', '\+']
                for mode in wrong_index:
                    wait_play[wait_play.str.contains(mode, na=False)] = np.nan
            except AttributeError:
                pass

            wait_play = wait_play.apply(nums_ave)

            try:
                wait_play = pd.to_numeric(wait_play)
            except ValueError:
                print('没处理好数据啊！！！！！')
                raise ValueError
            print('finished')

            hold_play.append(wait_play)

        except KeyError:
            print(index_name + ' no exist!')

    return hold_play
def clean_2690__(feature):
    index_format = '2690%02d'
    list_min = 3
    list_max = 23
    hold_play = []
    for index in range(list_min, list_max+1):
        index_name = index_format % index
        try:
            print('start ' + index_name)
            wait_play = feature[index_name]
            wait_play = wait_play.apply(strQ2B)
            wait_play = wait_play.apply(nums_ave)

            try:
                wrong_index = ['--', '未见']
                for mode in wrong_index:
                    wait_play[wait_play.str.contains(mode, na=False)] = np.nan

            except AttributeError:
                pass

            try:
                wait_play = pd.to_numeric(wait_play)
            except ValueError:
                print('没处理好数据啊！！！！！')
                raise ValueError
            print('finished')

            hold_play.append(wait_play)

        except KeyError:
            print(index_name + ' no exist!')

    return hold_play
def clean_3__(feature):
    index_format = '3%02d'
    in_list = [10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 60]
    hold_play = []
    for index in in_list:
        index_name = index_format % index
        try:
            print('start ' + index_name)
            wait_play = feature[index_name]
            wait_play = wait_play.apply(strQ2B)
            # wait_play = wait_play.apply(two_dot_replace)

            try:
                wrong_index = ['\+', '阴性']
                for mode in wrong_index:
                    wait_play[wait_play.str.contains(mode, na=False)] = np.nan
            except AttributeError:
                pass

            wait_play = wait_play.apply(nums_ave)

            try:
                wait_play = pd.to_numeric(wait_play)
            except ValueError:
                print('没处理好数据啊！！！！！')
                raise ValueError
            print('finished')

            hold_play.append(wait_play)

        except KeyError:
            print(index_name + ' no exist!')

    return hold_play
def clean_3000__(feature):
    index_format = '3000%02d'
    in_list = [1, 5, 7, 8, 9, 11, 12, 13, 14, 17, 18, 19, 21, 36, 68, 70, 74, 76, 92]
    hold_play = []
    for index in in_list:
        index_name = index_format % index
        try:
            print('start ' + index_name)
            wait_play = feature[index_name]
            wait_play = wait_play.apply(strQ2B)
            wait_play = wait_play.apply(nums_ave)

            try:
                wrong_index = ['--', '未见', '标本已退检', '\/',
                               '女性肿瘤指标', '阴性', '阳性', '\+',
                               '弱阳', 'A', '\-', '未检出', '正常',
                               '少', '满', '脓白', '偶', 'Ⅱ']
                for mode in wrong_index:
                    wait_play[wait_play.str.contains(mode, na=False)] = np.nan

            except AttributeError:
                pass

            try:
                wait_play = pd.to_numeric(wait_play)
            except ValueError:
                print('没处理好数据啊！！！！！')
                raise ValueError
            print('finished')

            hold_play.append(wait_play)

        except KeyError:
            print(index_name + ' no exist!')

    return hold_play
def clean_31__(feature):
    index_format = '31%02d'
    in_list = [84, 93]
    hold_play = []
    for index in in_list:
        index_name = index_format % index
        try:
            print('start ' + index_name)
            wait_play = feature[index_name]
            wait_play = wait_play.apply(strQ2B)
            # wait_play = wait_play.apply(two_dot_replace)

            try:
                wrong_index = ['阴性']
                for mode in wrong_index:
                    wait_play[wait_play.str.contains(mode, na=False)] = np.nan
            except AttributeError:
                pass

            wait_play = wait_play.apply(nums_ave)

            try:
                wait_play = pd.to_numeric(wait_play)
            except ValueError:
                print('没处理好数据啊！！！！！')
                raise ValueError
            print('finished')

            hold_play.append(wait_play)

        except KeyError:
            print(index_name + ' no exist!')

    return hold_play
def clean_31__2(feature):
    index_format = '31%02d'
    in_list = [90, 91, 92, 94, 95, 96, 97, 98]
    hold_play = []
    for index in in_list:
        index_name = index_format % index
        try:
            print('start ' + index_name)
            wait_play = feature[index_name]

            wait_play = wait_play.apply(strQ2B)
            wait_play = wait_play.apply(nums_ave)

            d = wait_play
            d = pd.to_numeric(d, errors='coerce')
            wait_play[~d.isna()] = np.nan

            wrong_index = ['\+\-', '3-5', '少数', '偶见', '未检出']
            for mode in wrong_index:
                wait_play[wait_play.str.contains(mode, na=False)] = np.nan

            wrong_index = ['\+']
            for mode in wrong_index:
                wait_play[wait_play.str.contains(mode, na=False)] = '阳性'

            wrong_index = ['\-']
            for mode in wrong_index:
                wait_play[wait_play.str.contains(mode, na=False)] = '阴性'

            symptom = [['阳性'],
                       ['阴性', '正常', 'Normal', 'NormaL']]
            for cls, mode_list in enumerate(symptom):
                for mode in mode_list:
                    wait_play[wait_play.str.contains(mode, na=False)] = '{}_'.format(index) + str(cls)

            wait_play = wait_play.fillna('{}_'.format(index) + str(len(symptom)))
            wait_play = pd.get_dummies(wait_play)
            wait_play = wait_play.drop(columns='{}_'.format(index) + str(len(symptom)))

            hold_play.append(wait_play)

        except KeyError:
            print(index_name + ' no exist!')
    return hold_play

def clean_6690__(feature):
    index_format = '6690%02d'
    in_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 21]
    hold_play = []
    for index in in_list:
        index_name = index_format % index
        try:
            print('start ' + index_name)
            wait_play = feature[index_name]
            wait_play = wait_play.apply(strQ2B)
            wait_play = wait_play.apply(nums_ave)

            try:
                wrong_index = ['--', '\/']
                for mode in wrong_index:
                    wait_play[wait_play.str.contains(mode, na=False)] = np.nan

            except AttributeError:
                pass

            try:
                wait_play = pd.to_numeric(wait_play)
            except ValueError:
                print('没处理好数据啊！！！！！')
                raise ValueError
            print('finished')

            hold_play.append(wait_play)

        except KeyError:
            print(index_name + ' no exist!')

    return hold_play
def clean_8090__(feature):
    index_format = '8090%02d'
    in_list = [1, 2, 3, 4, 7, 8, 9, 10, 13, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 29,
               31, 32, 33, 34, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
               52, 53, 54, 55, 56, 57, 58, 59, 60, 61]
    hold_play = []
    for index in in_list:
        index_name = index_format % index
        try:
            print('start ' + index_name)
            wait_play = feature[index_name]
            wait_play = wait_play.apply(strQ2B)
            wait_play = wait_play.apply(nums_ave)

            try:
                wrong_index = ['标本已退检', '\/', '阴性']
                for mode in wrong_index:
                    wait_play[wait_play.str.contains(mode, na=False)] = np.nan

            except AttributeError:
                pass

            try:
                wait_play = pd.to_numeric(wait_play)
            except ValueError:
                print('没处理好数据啊！！！！！')
                raise ValueError
            print('finished')

            hold_play.append(wait_play)

        except KeyError:
            print(index_name + ' no exist!')

    return hold_play
def clean_9790__(feature):
    index_format = '9790%02d'
    list_min = 1
    list_max = 23
    hold_play = []
    for index in range(list_min, list_max+1):
        index_name = index_format % index
        try:
            print('start ' + index_name)
            wait_play = feature[index_name]
            wait_play = wait_play.apply(strQ2B)
            wait_play = wait_play.apply(nums_ave)

            try:
                wrong_index = ['--', '未见']  # 979001 979002 979003
                for mode in wrong_index:
                    wait_play[wait_play.str.contains(mode, na=False)] = np.nan

            except AttributeError:
                pass

            try:
                wait_play = pd.to_numeric(wait_play)
            except ValueError:
                print('没处理好数据啊！！！！！')
                raise ValueError
            print('finished')

            hold_play.append(wait_play)

        except KeyError:
            print(index_name + ' no exist!')

    return hold_play

# old
def clean_0424(feature):
    print('start 0424')
    wait_play = feature['0424']

    wait_play = wait_play.apply(nums_ave)

    # wrong_index = ['次/分']
    # for mode in wrong_index:
    #     wait_play[wait_play.str.contains(mode, na=False)] = np.nan
    normal_index = ['未见异常', '心率正常', '正常']
    exceed_index = ['窦性心动过速']
    not_enough_index = ['窦性心动过缓', '心动过缓']
    for mode in normal_index:
        wait_play[wait_play.str.contains(mode, na=False)] = 80.
    for mode in exceed_index:
        wait_play[wait_play.str.contains(mode, na=False)] = 105.
    for mode in not_enough_index:
        wait_play[wait_play.str.contains(mode, na=False)] = 55.

    try:
        wait_play = pd.to_numeric(wait_play)
    except ValueError:
        print('没处理好数据啊！！！！！')
        raise ValueError
    print('finished')
    return wait_play
def clean_30007(feature):
    print('start 30007')
    wait_play = feature['30007']

    not_know_index = ['见TCT', 'yellow', '阴性', '微混', '见刮片', '\+']
    for mode in not_know_index:
        wait_play[wait_play.str.contains(mode, na=False)] = np.nan

    ok_index = ['正常', '未见异常']
    for mode in ok_index:
        wait_play[wait_play.str.contains(mode, na=False)] = '30007_0'

    l4_index = ['Ⅰv', 'iV', 'iv', 'IV', 'Ⅳ']
    for mode in l4_index:
        wait_play[wait_play.str.contains(mode, na=False)] = '30007_4'

    l3_index = ['iii', 'III', 'Ⅲ']
    for mode in l3_index:
        wait_play[wait_play.str.contains(mode, na=False)] = '30007_3'

    l2_index = ['Ⅱ', 'ii', 'II', '中度']
    for mode in l2_index:
        wait_play[wait_play.str.contains(mode, na=False)] = '30007_2'

    l1_index = ['Ⅰ', 'i', 'I']
    for mode in l1_index:
        wait_play[wait_play.str.contains(mode, na=False)] = '30007_1'

    wait_play = wait_play.fillna('30007_5')

    wait_play = pd.get_dummies(wait_play)
    wait_play = wait_play.drop(columns='30007_5')
    print('finished')
    return wait_play
def clean_100006(feature):
    print('start 100006')
    wait_play = feature['100006']
    wait_play = wait_play.apply(out_other)

    wrong_index = ['---']
    for mode in wrong_index:
        wait_play[wait_play.str.contains(mode, na=False)] = np.nan

    try:
        wait_play = pd.to_numeric(wait_play) # , errors='coerce')
    except ValueError:
        print('没处理好数据啊！！！！！')
        raise ValueError

    print('finished')
    return wait_play
def clean_2174(feature):
    print('start 2174')
    wait_play = feature['2174']
    wait_play = wait_play.apply(strQ2B)

    try:
        wait_play = pd.to_numeric(wait_play)
    except ValueError:
        print('没处理好数据啊！！！！！')
        raise ValueError

    print('finished')
    return wait_play
def clean_183(feature):
    print('start 183')
    wait_play = feature['183']
    wait_play = wait_play.apply(strQ2B)

    wrong_index = ['77..21']  # p=re.search('[0-9]+\.\.[0-9]', s) 你看看怎么匹配
    for mode in wrong_index:
        wait_play[wait_play.str.contains(mode, na=False)] = '77.21'

    try:
        wait_play = pd.to_numeric(wait_play)
    except ValueError:
        print('没处理好数据啊！！！！！')
        raise ValueError

    print('finished')
    return wait_play

def clean_190(feature):
    print('start 190')
    wait_play = feature['190']
    wait_play = wait_play.apply(strQ2B)

    try:
        wait_play = pd.to_numeric(wait_play)
    except ValueError:
        print('没处理好数据啊！！！！！')
        raise ValueError

    print('finished')
    return wait_play
def clean_191(feature):
    print('start 191')
    wait_play = feature['191']
    wait_play = wait_play.apply(strQ2B)

    try:
        wait_play = pd.to_numeric(wait_play)
    except ValueError:
        print('没处理好数据啊！！！！！')
        raise ValueError

    print('finished')
    return wait_play
def clean_192(feature):
    print('start 192')
    wait_play = feature['192']
    wait_play = wait_play.apply(strQ2B)

    wrong_index = ['12.01.']
    for mode in wrong_index:
        wait_play[wait_play.str.contains(mode, na=False)] = '12.01'

    wrong_index = ['16.7.07']
    for mode in wrong_index:
        wait_play[wait_play.str.contains(mode, na=False)] = '16.7'

    try:
        wait_play = pd.to_numeric(wait_play)
    except ValueError:
        print('没处理好数据啊！！！！！')
        raise ValueError

    print('finished')
    return wait_play
def clean_193(feature):
    print('start 193')
    wait_play = feature['193']
    wait_play = wait_play.apply(strQ2B)

    label_index = ['<']
    for mode in label_index:
        wait_play[wait_play.str.contains(mode, na=False)] = 1.8

    try:
        wait_play = pd.to_numeric(wait_play)
    except ValueError:
        print('没处理好数据啊！！！！！')
        raise ValueError

    print('finished')
    return wait_play

def clean_31(feature):
    print('start 31')
    wait_play = feature['31']
    wait_play = wait_play.apply(strQ2B)

    label_index = ['<']
    for mode in label_index:
        wait_play[wait_play.str.contains(mode, na=False)] = '0.6'

    wrong_index = ['5.10.']
    for mode in wrong_index:
        wait_play[wait_play.str.contains(mode, na=False)] = '5.10'

    try:
        wait_play = pd.to_numeric(wait_play)
    except ValueError:
        print('没处理好数据啊！！！！！')
        raise ValueError

    print('finished')
    return wait_play
def clean_314(feature):
    print('start 314')
    wait_play = feature['314']
    wait_play = wait_play.apply(strQ2B)

    wrong_index = ['标本已退检']
    for mode in wrong_index:
        wait_play[wait_play.str.contains(mode, na=False)] = np.nan

    try:
        wait_play = pd.to_numeric(wait_play)
    except ValueError:
        print('没处理好数据啊！！！！！')
        raise ValueError

    print('finished')
    return wait_play
def clean_32(feature):
    print('start 32')
    wait_play = feature['32']
    wait_play = wait_play.apply(strQ2B)

    wrong_index = ['2.1.']
    for mode in wrong_index:
        wait_play[wait_play.str.contains(mode, na=False)] = 2.1

    try:
        wait_play = pd.to_numeric(wait_play)
    except ValueError:
        print('没处理好数据啊！！！！！')
        raise ValueError

    print('finished')
    return wait_play
def clean_320(feature):
    print('start 320')
    wait_play = feature['320']
    wait_play = wait_play.apply(strQ2B)

    wrong_index = ['---']
    for mode in wrong_index:
        wait_play[wait_play.str.contains(mode, na=False)] = np.nan

    try:
        wait_play = pd.to_numeric(wait_play)
    except ValueError:
        print('没处理好数据啊！！！！！')
        raise ValueError

    print('finished')
    return wait_play
def clean_38(feature):
    print('start 38')
    wait_play = feature['38']
    wait_play = wait_play.apply(strQ2B)

    wrong_index = ['8.53.']
    for mode in wrong_index:
        wait_play[wait_play.str.contains(mode, na=False)] = 8.53

    try:
        wait_play = pd.to_numeric(wait_play)
    except ValueError:
        print('没处理好数据啊！！！！！')
        raise ValueError

    print('finished')
    return wait_play
def clean_312(feature):
    print('start 312')
    wait_play = feature['312']
    wait_play = wait_play.apply(strQ2B)
    wait_play = wait_play.apply(nums_ave)

    try:
        wait_play = pd.to_numeric(wait_play)
    except ValueError:
        print('没处理好数据啊！！！！！')
        raise ValueError

    print('finished')
    return wait_play
def clean_313(feature):
    print('start 313')
    wait_play = feature['313']
    wait_play = wait_play.apply(strQ2B)
    wait_play = wait_play.apply(nums_ave)

    try:
        wait_play = pd.to_numeric(wait_play)
    except ValueError:
        print('没处理好数据啊！！！！！')
        raise ValueError

    print('finished')
    return wait_play
def clean_155(feature):
    print('start 155')
    wait_play = feature['155']
    wait_play = wait_play.apply(strQ2B)
    wait_play = wait_play.apply(nums_ave)

    try:
        wait_play = pd.to_numeric(wait_play)
    except ValueError:
        print('没处理好数据啊！！！！！')
        raise ValueError

    print('finished')
    return wait_play
def clean_1321(feature):
    print('start 1321')
    wait_play = feature['1321']

    # 症状表
    symptom = [['光感'],
               ['失明', '义眼', '无光感'],
               ['正常'],
               ['因无法配合不能检测']]
    symptom_value = ['0.01',
                     '0.',
                     '1.1',
                     np.nan]
    for cls, mode_list in enumerate(symptom):
        for mode in mode_list:
            wait_play[wait_play.str.contains(mode, na=False)] = symptom_value[cls]

    wait_play = wait_play.apply(nums_ave)

    try:
        wait_play = pd.to_numeric(wait_play)
    except ValueError:
        print('没处理好数据啊！！！！！')
        raise ValueError

    print('finished')
    return wait_play
def clean_1322(feature):
    print('start 1322')
    wait_play = feature['1322']

    # 症状表
    symptom = [['光感'],
               ['失明', '义眼', '无光感'],
               ['正常'],
               ['因无法配合不能检测', '指数,建议医院进一步检查']]
    symptom_value = ['0.01',
                     '0.',
                     '1.1',
                     np.nan]
    for cls, mode_list in enumerate(symptom):
        for mode in mode_list:
            wait_play[wait_play.str.contains(mode, na=False)] = symptom_value[cls]

    try:
        wait_play = pd.to_numeric(wait_play)
    except ValueError:
        print('没处理好数据啊！！！！！')
        raise ValueError

    print('finished')
    return wait_play

def clean_0202(feature):
    print('start 0202')
    wait_play = feature['0202']

    label_index = ['手术后', 'exit']
    for mode in label_index:
        wait_play[wait_play.str.contains(mode, na=False)] = np.nan

    symptom = [['未见异常', '未见明显异常', '正常'],
               ['耵聍', '栓塞'],
               ['红肿', '湿疹', '炎', '霉苔', '干燥', '疖肿', '分泌物', '轻度充血'],  # 炎症
               ['狭窄', '扩大', '闭锁', '痣', '异物', '赘生物'],  # 结构性改变
               ['血痂', '出血点', '新生物'],  # 其他
               ['助听器']]
    for cls, mode_list in enumerate(symptom):
        for mode in mode_list:
            wait_play[wait_play.str.contains(mode, na=False)] = '0202_' + str(cls)

    wait_play = wait_play.fillna('0202_' + str(len(symptom)))
    wait_play = pd.get_dummies(wait_play)
    wait_play = wait_play.drop(columns='0202_' + str(len(symptom)))
    print('finished')
    return wait_play
def clean_0206(feature):
    print('start 0206')
    wait_play = feature['0206']

    label_index = ['exit']
    for mode in label_index:
        wait_play[wait_play.str.contains(mode, na=False)] = np.nan

    symptom = [['未见异', '未见明显异常', '正常', '无压痛'],
               ['双侧乳突压痛'],
               ['右乳突压痛'],
               ['右耳乳突根治术后', '右乳突炎手术'],
               ['左耳乳突根治术后', '左乳突炎手术']]
    for cls, mode_list in enumerate(symptom):
        for mode in mode_list:
            wait_play[wait_play.str.contains(mode, na=False)] = '0206_' + str(cls)

    wait_play = wait_play.fillna('0206_' + str(len(symptom)))
    wait_play = pd.get_dummies(wait_play)
    wait_play = wait_play.drop(columns='0206_' + str(len(symptom)))
    print('finished')
    return wait_play
def clean_0207(feature):
    print('start 0207')
    wait_play = feature['0207']

    label_index = ['exit']
    for mode in label_index:
        wait_play[wait_play.str.contains(mode, na=False)] = np.nan

    symptom = [['未见异常', '未见明显异常', '正常'],
               ['鼻部肿物', '湿疹', '酒糟鼻', '鼻中隔右偏', '鼻甲肥大', '前庭']]
    for cls, mode_list in enumerate(symptom):
        for mode in mode_list:
            wait_play[wait_play.str.contains(mode, na=False)] = '0207_' + str(cls)

    wait_play = wait_play.fillna('0207_' + str(len(symptom)))
    wait_play = pd.get_dummies(wait_play)
    wait_play = wait_play.drop(columns='0207_' + str(len(symptom)))
    print('finished')
    return wait_play
def clean_0212(feature):
    print('start 0212')
    wait_play = feature['0212']

    symptom = [['未见异常', '未见明显异常', '正常', '无压痛'],
               ['炎'],
               ['压痛']]
    for cls, mode_list in enumerate(symptom):
        for mode in mode_list:
            wait_play[wait_play.str.contains(mode, na=False)] = '0212_' + str(cls)

    wait_play = wait_play.fillna('0212_' + str(len(symptom)))
    wait_play = pd.get_dummies(wait_play)
    wait_play = wait_play.drop(columns='0212_' + str(len(symptom)))
    print('finished')
    return wait_play
def clean_0213(feature):
    print('start 0213')
    wait_play = feature['0213']

    label_index = ['视不见', '分泌物', '肥大']
    for mode in label_index:
        wait_play[wait_play.str.contains(mode, na=False)] = np.nan

    symptom = [['未见异常', '未见明显异常', '正常'],
               ['鼻炎'],
               ['鼻粘膜', '鼻黏膜']]
    for cls, mode_list in enumerate(symptom):
        for mode in mode_list:
            wait_play[wait_play.str.contains(mode, na=False)] = '0213_' + str(cls)

    wait_play = wait_play.fillna('0213_' + str(len(symptom)))
    wait_play = pd.get_dummies(wait_play)
    wait_play = wait_play.drop(columns='0213_' + str(len(symptom)))
    print('finished')
    return wait_play
def clean_0216(feature):
    print('start 0216')
    wait_play = feature['0216']

    label_index = ['exit', '唇裂']
    for mode in label_index:
        wait_play[wait_play.str.contains(mode, na=False)] = np.nan

    symptom = [['未见异常', '未见明显异常', '正常'],
               ['充血', '水肿'],
               ['悬雍垂肥大', '悬雍垂部分切除', '悬雍垂切除'],
               ['息肉', '赘生物', '淡红色颗粒状突起', '乳头状瘤', '新生物'],
               ['过长'],
               ['畸形']]
    for cls, mode_list in enumerate(symptom):
        for mode in mode_list:
            wait_play[wait_play.str.contains(mode, na=False)] = '0216_' + str(cls)

    wait_play = wait_play.fillna('0216_' + str(len(symptom)))
    wait_play = pd.get_dummies(wait_play)
    wait_play = wait_play.drop(columns='0216_' + str(len(symptom)))
    print('finished')
    return wait_play
def clean_3399(feature):
    print('start 3399')
    wait_play = feature['3399']

    symptom = [['淡', '浅', '黄色', 'yellow'],
               ['红色', '褐色', '无色', '混浊', '其他', '深']]
    for cls, mode_list in enumerate(symptom):
        for mode in mode_list:
            wait_play[wait_play.str.contains(mode, na=False)] = '3399_' + str(cls)

    wait_play = wait_play.fillna('3399_' + str(len(symptom)))
    wait_play = pd.get_dummies(wait_play)
    wait_play = wait_play.drop(columns='3399_' + str(len(symptom)))
    print('finished')
    return wait_play
def clean_3400(feature):
    print('start 3400')
    wait_play = feature['3400']

    not_know_index = ['6.5', '\+']
    for mode in not_know_index:
        wait_play[wait_play.str.contains(mode, na=False)] = np.nan

    symptom = [['透明'],
               ['混浊', '浑浊'],
               ['微混', '微浑']]
    for cls, mode_list in enumerate(symptom):
        for mode in mode_list:
            wait_play[wait_play.str.contains(mode, na=False)] = '3400_' + str(cls)
    wait_play = wait_play.fillna('3400_' + str(len(symptom)))
    wait_play = pd.get_dummies(wait_play)
    wait_play = wait_play.drop(columns='3400_' + str(len(symptom)))
    print('finished')
    return wait_play
def clean_3601(feature):
    print('start 3601')
    wait_play = feature['3601']

    d = wait_play
    d = pd.to_numeric(d, errors='coerce')
    wait_play[~d.isna()] = np.nan

    not_know_index = ['T']
    for mode in not_know_index:
        wait_play[wait_play.str.contains(mode, na=False)] = np.nan

    symptom = [['骨量基本正常', '骨量正常', '正常', '未见异常'],
               ['骨量减少'],
               ['骨质疏松', '骨质减少', '骨密度降低']]
    for cls, mode_list in enumerate(symptom):
        for mode in mode_list:
            wait_play[wait_play.str.contains(mode, na=False)] = '3601_' + str(cls)

    wait_play = wait_play.fillna('3601_' + str(len(symptom)))
    wait_play = pd.get_dummies(wait_play)
    wait_play = wait_play.drop(columns='3601_' + str(len(symptom)))
    print('finished')
    return wait_play
def clean_0901(feature):
    print('start 0901')
    wait_play = feature['0901']

    not_know_index = ['瘢痕', '放疗痕', '术后', '黑痣', '带状疱疹', '斑秃', '脂肪瘤']
    for mode in not_know_index:
        wait_play[wait_play.str.contains(mode, na=False)] = np.nan

    not_know_index = ['0无', '见异常']
    wait_play[wait_play.isin(not_know_index)] = np.nan

    symptom = [['正常', '未见异常', '未见明显异常', '正常', '无特殊记载'],
               ['白癜风'],
               ['纹身'],
               ['皮肤白斑'],
               ['皮肤苍白'],
               ['色素沉着'],
               ['疤痕']]
    for cls, mode_list in enumerate(symptom):
        for mode in mode_list:
            wait_play[wait_play.str.contains(mode, na=False)] = '0901_' + str(cls)

    wait_play = wait_play.fillna('0901_' + str(len(symptom)))
    wait_play = pd.get_dummies(wait_play)
    wait_play = wait_play.drop(columns='0901_' + str(len(symptom)))
    print('finished')
    return wait_play

def clean_0981(feature):
    print('start 0981')
    wait_play = feature['0981']

    symptom = [['自述不查', '包皮弃查', '未发现明显异常', '未见异常'],
               ['过长', '包茎']]
    for cls, mode_list in enumerate(symptom):
        for mode in mode_list:
            wait_play[wait_play.str.contains(mode, na=False)] = '0981_' + str(cls)

    wait_play = wait_play.fillna('0981_' + str(len(symptom)))
    wait_play = pd.get_dummies(wait_play)
    wait_play = wait_play.drop(columns='0981_' + str(len(symptom)))
    print('finished')
    return wait_play
def clean_0983(feature):
    print('start 0983')
    wait_play = feature['0983']

    symptom = [['自述不查', '未见异常', '未发现明显异常', '未发现异常'],
               ['双睾切除', '单侧睾丸', '睾丸缺如', '睾丸稍增大', '睾丸肿大', '睾丸偏小', '右侧睾丸为左侧睾丸大小的1/3', '附睾肿大'],
               ['结节', '肿物'],
               ['阴囊皮肤'],
               ['精索静脉', '精索囊肿'],
               ['鞘膜积液']]
    for cls, mode_list in enumerate(symptom):
        for mode in mode_list:
            wait_play[wait_play.str.contains(mode, na=False)] = '0983_' + str(cls)

    wait_play = wait_play.fillna('0983_' + str(len(symptom)))
    wait_play = pd.get_dummies(wait_play)
    wait_play = wait_play.drop(columns='0983_' + str(len(symptom)))
    print('finished')
    return wait_play
def clean_0406(feature):
    print('start 0406')
    wait_play = feature['0406']

    not_know_index = ['不满意']
    for mode in not_know_index:
        wait_play[wait_play.str.contains(mode, na=False)] = np.nan

    not_know_index = ['肋下约1cm', '有移动性浊音']
    wait_play[wait_play.isin(not_know_index)] = np.nan

    symptom = [['未触及', '未及', '未见异常', '正常', '不大'],
               ['肝肿大', '可触及']]
    for cls, mode_list in enumerate(symptom):
        for mode in mode_list:
            wait_play[wait_play.str.contains(mode, na=False)] = '0406_' + str(cls)

    wait_play = wait_play.fillna('0406_' + str(len(symptom)))
    wait_play = pd.get_dummies(wait_play)
    wait_play = wait_play.drop(columns='0406_' + str(len(symptom)))
    print('finished')
    return wait_play
def clean_0407(feature):
    print('start 0407')
    wait_play = feature['0407']

    not_know_index = ['不满意', '脾脏有压痛']
    for mode in not_know_index:
        wait_play[wait_play.str.contains(mode, na=False)] = np.nan

    not_know_index = ['未 触及']
    wait_play[wait_play.isin(not_know_index)] = '未触及'

    symptom = [['未触及', '未及', '未见异常', '正常', '不大'],
               ['脾肿大', '可触及'],
               ['脾脏切除']]
    for cls, mode_list in enumerate(symptom):
        for mode in mode_list:
            wait_play[wait_play.str.contains(mode, na=False)] = '0407_' + str(cls)

    wait_play = wait_play.fillna('0407_' + str(len(symptom)))
    wait_play = pd.get_dummies(wait_play)
    wait_play = wait_play.drop(columns='0407_' + str(len(symptom)))
    print('finished')
    return wait_play
def clean_0415(feature):
    print('start 0415')
    wait_play = feature['0415']

    symptom = [['未见异常', '未见明显异常', '无神经定位体征'],
               ['左肢肌力减弱'],
               ['右上下肢肌力减弱']]
    for cls, mode_list in enumerate(symptom):
        for mode in mode_list:
            wait_play[wait_play.str.contains(mode, na=False)] = '0415_' + str(cls)

    wait_play = wait_play.fillna('0415_' + str(len(symptom)))
    wait_play = pd.get_dummies(wait_play)
    wait_play = wait_play.drop(columns='0415_' + str(len(symptom)))
    print('finished')
    return wait_play
def clean_0420(feature):
    print('start 0420')
    wait_play = feature['0420']

    label_index = ['.........', '第二心音分裂', '主动脉第2心音强', '右位心']
    for mode in label_index:
        wait_play[wait_play.str.contains(mode, na=False)] = np.nan

    symptom = [['正常', '未见异常', '未闻及异常', '未见明显异常', '有力', '心音强'],
               ['强弱不等'],
               ['弱', '较低', '低钝', '心音弱'],
               ['心音遥远']
               ]
    for cls, mode_list in enumerate(symptom):
        for mode in mode_list:
            wait_play[wait_play.str.contains(mode, na=False)] = '0420_' + str(cls)

    wait_play = wait_play.fillna('0420_' + str(len(symptom)))
    wait_play = pd.get_dummies(wait_play)
    wait_play = wait_play.drop(columns='0420_' + str(len(symptom)))
    print('finished')
    return wait_play
def clean_0423(feature):
    print('start 0423')
    wait_play = feature['0423']

    symptom = [['正常', '未见异常', '未见明显异常', '清'],
               ['粗', '弱', '消失', '哮鸣']]
    for cls, mode_list in enumerate(symptom):
        for mode in mode_list:
            wait_play[wait_play.str.contains(mode, na=False)] = '0423_' + str(cls)

    wait_play = wait_play.fillna('0423_' + str(len(symptom)))
    wait_play = pd.get_dummies(wait_play)
    wait_play = wait_play.drop(columns='0423_' + str(len(symptom)))
    print('finished')
    return wait_play
def clean_0427(feature):
    index = '0427'
    print('start {}'.format(index))
    wait_play = feature[index]

    d = wait_play
    d = pd.to_numeric(d, errors='coerce')
    wait_play[~d.isna()] = np.nan

    symptom = [['未见异常', '未见明显异常', '无重大手术记载', '无重大手术史', '正常'],
               ['奔马律'],
               ['法乐氏四联症手术', '心脏射频消融手术'],
               ['扩心病']]
    for cls, mode_list in enumerate(symptom):
        for mode in mode_list:
            wait_play[wait_play.str.contains(mode, na=False)] = '{}_'.format(index) + str(cls)

    wait_play = wait_play.fillna('{}_'.format(index) + str(len(symptom)))
    wait_play = pd.get_dummies(wait_play)
    wait_play = wait_play.drop(columns='{}_'.format(index) + str(len(symptom)))
    print('finished')
    return wait_play
def clean_0430(feature):
    print('start 0430')
    wait_play = feature['0430']

    symptom = [['未触及', '未见异常', '正常', '软'],
               ['中', '硬']]
    for cls, mode_list in enumerate(symptom):
        for mode in mode_list:
            wait_play[wait_play.str.contains(mode, na=False)] = '0430_' + str(cls)

    wait_play = wait_play.fillna('0430_' + str(len(symptom)))
    wait_play = pd.get_dummies(wait_play)
    wait_play = wait_play.drop(columns='0430_' + str(len(symptom)))
    print('finished')
    return wait_play
def clean_0431(feature):
    print('start 0431')
    wait_play = feature['0431']

    not_know_index = ['胆囊切除术后']
    wait_play[wait_play.isin(not_know_index)] = np.nan

    symptom = [['无压痛点', '未见异常', '未触及', '正常'],
               ['叩击痛'],
               ['压痛']]
    for cls, mode_list in enumerate(symptom):
        for mode in mode_list:
            wait_play[wait_play.str.contains(mode, na=False)] = '0431_' + str(cls)

    wait_play = wait_play.fillna('0431_' + str(len(symptom)))
    wait_play = pd.get_dummies(wait_play)
    wait_play = wait_play.drop(columns='0431_' + str(len(symptom)))
    print('finished')
    return wait_play

def clean_0440(feature):
    print('start 0440')
    wait_play = feature['0440']

    d = wait_play
    d = pd.to_numeric(d, errors='coerce')
    wait_play[~d.isna()] = np.nan

    label_index = ['起止日期']
    for mode in label_index:
        wait_play[wait_play.str.contains(mode, na=False)] = np.nan

    symptom = [['肾区无叩痛', '未见异常', '未触及', '正常', '无重大手术史', '双肾无叩痛'],
               ['左肾有叩痛'],
               ['右肾有叩痛'],
               ['双肾有叩痛']]
    for cls, mode_list in enumerate(symptom):
        for mode in mode_list:
            wait_play[wait_play.str.contains(mode, na=False)] = '0440_' + str(cls)

    wait_play = wait_play.fillna('0440_' + str(len(symptom)))
    wait_play = pd.get_dummies(wait_play)
    wait_play = wait_play.drop(columns='0440_' + str(len(symptom)))
    print('finished')
    return wait_play
def clean_0728(feature):
    print('start 0728')
    wait_play = feature['0728']

    symptom = [['未见异常', '未见明显异常'],
               ['双侧腮导口充血'],
               ['左侧良性肿大']]
    for cls, mode_list in enumerate(symptom):
        for mode in mode_list:
            wait_play[wait_play.str.contains(mode, na=False)] = '0728_' + str(cls)

    wait_play = wait_play.fillna('0728_' + str(len(symptom)))
    wait_play = pd.get_dummies(wait_play)
    wait_play = wait_play.drop(columns='0728_' + str(len(symptom)))
    print('finished')
    return wait_play
def clean_0732(feature):
    print('start 0732')
    wait_play = feature['0732']

    symptom = [['未见异常', '未见明显异常', '正常'],
               ['沟纹舌'],
               ['毛舌'],
               ['萎缩性舌炎']]
    for cls, mode_list in enumerate(symptom):
        for mode in mode_list:
            wait_play[wait_play.str.contains(mode, na=False)] = '0732_' + str(cls)

    wait_play = wait_play.fillna('0732_' + str(len(symptom)))
    wait_play = pd.get_dummies(wait_play)
    wait_play = wait_play.drop(columns='0732_' + str(len(symptom)))
    print('finished')
    return wait_play
def clean_1328(feature):
    print('start 1328')
    wait_play = feature['1328']

    symptom = [['正常', '未见异常', '未发现明显异常'],
               ['色弱'],
               ['色盲']]
    for cls, mode_list in enumerate(symptom):
        for mode in mode_list:
            wait_play[wait_play.str.contains(mode, na=False)] = '1328_' + str(cls)

    wait_play = wait_play.fillna('1328_' + str(len(symptom)))
    wait_play = pd.get_dummies(wait_play)
    wait_play = wait_play.drop(columns='1328_' + str(len(symptom)))
    print('finished')
    return wait_play
def clean_100014(feature):
    print('start 100014')
    wait_play = feature['100014']

    wait_play = wait_play.apply(strQ2B)

    not_know_index = ['4.3.']
    for mode in not_know_index:
        wait_play[wait_play.str.contains(mode, na=False)] = 4.3

    not_know_index = ['20.908.']
    for mode in not_know_index:
        wait_play[wait_play.str.contains(mode, na=False)] = 20.908

    not_know_index = ['>100']
    for mode in not_know_index:
        wait_play[wait_play.str.contains(mode, na=False)] = 101

    try:
        wait_play = pd.to_numeric(wait_play)
    except ValueError:
        print('没处理好数据啊！！！！！')
        raise ValueError
    print('finished')
    return wait_play
def clean_1325(feature):
    print('start 1325')
    wait_play = feature['1325']

    wait_play = wait_play.apply(strQ2B)

    not_know_index = ['正常', '义眼', '失明', '光感']
    for mode in not_know_index:
        wait_play[wait_play.str.contains(mode, na=False)] = np.nan

    not_know_index = [',']
    for mode in not_know_index:
        wait_play[wait_play.str.contains(mode, na=False)] = np.nan

    try:
        wait_play = pd.to_numeric(wait_play)
    except ValueError:
        print('没处理好数据啊！！！！！')
        raise ValueError
    print('finished')
    return wait_play
def clean_1326(feature):
    print('start 1326')
    wait_play = feature['1326']

    wait_play = wait_play.apply(strQ2B)

    not_know_index = ['正常', '义眼', '失明', '光感']
    for mode in not_know_index:
        wait_play[wait_play.str.contains(mode, na=False)] = np.nan

    not_know_index = [',']
    for mode in not_know_index:
        wait_play[wait_play.str.contains(mode, na=False)] = np.nan

    try:
        wait_play = pd.to_numeric(wait_play)
    except ValueError:
        print('没处理好数据啊！！！！！')
        raise ValueError
    print('finished')
    return wait_play
def clean_2409(feature):
    print('start 2409')
    wait_play = feature['2409']

    wait_play = wait_play.apply(strQ2B)

    not_know_index = ['阴性', '\+']
    for mode in not_know_index:
        wait_play[wait_play.str.contains(mode, na=False)] = None

    not_know_index = ['<200']
    for mode in not_know_index:
        wait_play[wait_play.str.contains(mode, na=False)] = 195

    not_know_index = ['<50']
    for mode in not_know_index:
        wait_play[wait_play.str.contains(mode, na=False)] = 48

    not_know_index = ['<20']
    for mode in not_know_index:
        wait_play[wait_play.str.contains(mode, na=False)] = 18

    not_know_index = ['<16']
    for mode in not_know_index:
        wait_play[wait_play.str.contains(mode, na=False)] = 14

    not_know_index = ['<10.0']
    for mode in not_know_index:
        wait_play[wait_play.str.contains(mode, na=False)] = 8

    not_know_index = ['<4']
    for mode in not_know_index:
        wait_play[wait_play.str.contains(mode, na=False)] = 2

    not_know_index = ['>201']
    for mode in not_know_index:
        wait_play[wait_play.str.contains(mode, na=False)] = 205

    not_know_index = ['正常值']  # 2409
    for mode in not_know_index:
        wait_play[wait_play.str.contains(mode, na=False)] = wait_play.str[0:4]

    not_know_index = ['25%']  # 2409
    for mode in not_know_index:
        wait_play[wait_play.str.contains(mode, na=False)] = 25

    try:
        wait_play = pd.to_numeric(wait_play)
    except ValueError:
        print('没处理好数据啊！！！！！')
        raise ValueError
    print('finished')
    return wait_play
def clean_0421(feature):
    print('start 0421')
    wait_play = feature['0421']

    d = wait_play
    d = pd.to_numeric(d, errors='coerce')
    wait_play[~d.isna()] = np.nan

    label_index = ['整']
    for mode in label_index:
        wait_play[wait_play.str.contains(mode, na=False)] = '整齐'

    label_index = ['窦性心律不齐']
    for mode in label_index:
        wait_play[wait_play.str.contains(mode, na=False)] = '窦不'

    label_index = ['早搏']
    for mode in label_index:
        wait_play[wait_play.str.contains(mode, na=False)] = '早搏'

    label_index = ['房颤', '心律绝对不齐', '绝对不规则']
    for mode in label_index:
        wait_play[wait_play.str.contains(mode, na=False)] = '房颤或不规则'

    label_index = ['不齐', '过']
    for mode in label_index:
        wait_play[wait_play.str.contains(mode, na=False)] = '心律不正'

    label_index = ['窦性心律']
    for mode in label_index:
        wait_play[wait_play.str.contains(mode, na=False)] = np.nan

    symptom = [['正常', '未见异常', '整齐', '齐'],
               ['心律不正', '房颤或不规则', '早搏', '窦不']]
    for cls, mode_list in enumerate(symptom):
        for mode in mode_list:
            wait_play[wait_play.str.contains(mode, na=False)] = '0421_' + str(cls)

    wait_play = wait_play.fillna('0421_' + str(len(symptom)))
    wait_play = pd.get_dummies(wait_play)
    wait_play = wait_play.drop(columns='0421_' + str(len(symptom)))
    print('finished')
    return wait_play

def clean_0434(feature):
    print('start 0434')
    wait_play = feature['0434']

    not_know_index = ['未见异常', '无特殊记载', '健康']
    for mode in not_know_index:
        wait_play[wait_play.str.contains(mode, na=False)] = '04341'

    not_know_index = ['糖尿病']
    for mode in not_know_index:
        wait_play[wait_play.str.contains(mode, na=False)] = '04342'

    not_know_index = ['高血压', '血压偏高']
    for mode in not_know_index:
        wait_play[wait_play.str.contains(mode, na=False)] = '04343'

    not_know_index = ['高血脂', '血脂偏高']
    for mode in not_know_index:
        wait_play[wait_play.str.contains(mode, na=False)] = '04344'

    not_know_index = ['脂肪肝']
    for mode in not_know_index:
        wait_play[wait_play.str.contains(mode, na=False)] = '04345'

    d = wait_play
    d = pd.to_numeric(d, errors='coerce')
    wait_play[d.isna()] = None

    wait_play = pd.get_dummies(wait_play)
    print('finished')
    return wait_play

def clean_0439(feature):
    print('start 0439')
    wait_play = feature['0439']

    not_know_index = ['cm', 'CM', '68', '99%', '血脂偏高']
    for mode in not_know_index:
        wait_play[wait_play.str.contains(mode, na=False)] = None

    symptom = [['未见异常'],
               ['父高血压', '父亲高血压', '父亲有高血压', '母高血压', '母亲高血压', '父母高血压', '父亲母亲高血压', '父母亲高血压''哥哥姐姐有高血压'],
               ['父糖尿病', '父亲糖尿病', '母糖尿病', '母亲糖尿病', '父母糖尿病', '父亲母亲糖尿病', '父母亲糖尿病', '奶奶、爸爸有糖尿病史'],
               ['高血压'],
               ['糖尿病'],
               ['癌'],
               ['冠心病', '梗']]
    for cls, mode_list in enumerate(symptom):
        for mode in mode_list:
            wait_play[wait_play.str.contains(mode, na=False)] = '0439_' + str(cls)

    wait_play = wait_play.fillna('0439_' + str(len(symptom)))
    wait_play = pd.get_dummies(wait_play)
    wait_play = wait_play.drop(columns='0439_' + str(len(symptom)))
    print('finished')
    return wait_play

def clean_BMI(feature):
    print('start BMI')
    wait_play1 = feature['2403']
    wait_play2 = feature['2404']
    wait_play3 = feature['2405']

    wait_play1[wait_play1 < 20] = None
    wait_play1[wait_play1 > 200] = None

    wait_play3 = 10000 * wait_play1 / wait_play2 / wait_play2

    hold_list = []
    hold_list.append(wait_play1)
    hold_list.append(wait_play2)
    hold_list.append(wait_play3)
    print('finished')
    return hold_list

def clean_0425_f(feature):
    print('start 0425 fenlei')
    feature = feature.copy()
    wait_play = feature['0425']

    label_index = ['分', '/']
    for mode in label_index:
        wait_play[wait_play.str.contains(mode, na=False)] = wait_play.str[0:2]

    label_index = ['26', '25', '24', '23', '22', '21']
    for mode in label_index:
        wait_play[wait_play.str.contains(mode, na=False)] = '急促'

    label_index = ['20', '19', '18', '17', '16', '15', '14', '13', '12']
    for mode in label_index:
        wait_play[wait_play.str.contains(mode, na=False)] = '正常'

    label_index = ['11', '10', '9', '8', '7']
    for mode in label_index:
        wait_play[wait_play.str.contains(mode, na=False)] = '缓慢'

    symptom = [['正常', '未见异常', '无异常', '未见明显异常', '清'],
               ['粗糙'],
               ['急促'],
               ['缓慢']]
    for cls, mode_list in enumerate(symptom):
        for mode in mode_list:
            wait_play[wait_play.str.contains(mode, na=False)] = '0425f_' + str(cls)

    wait_play = wait_play.fillna('0425f_' + str(len(symptom)))
    wait_play = pd.get_dummies(wait_play)
    wait_play = wait_play.drop(columns='0425f_' + str(len(symptom)))
    print('finished')
    return wait_play
def clean_0425_s(feature):
    print('start 0425 shuzhi')
    feature = feature.copy()
    wait_play = feature['0425']

    label_index = ['正常', '未见异常', '无异常', '未见明显异常']
    for mode in label_index:
        wait_play[wait_play.str.contains(mode, na=False)] = None

    label_index = ['分', '/']
    for mode in label_index:
        wait_play[wait_play.str.contains(mode, na=False)] = wait_play.str[0:2]

    label_index = ['粗糙', '缓慢', '急促', '清']
    for mode in label_index:
        wait_play[wait_play.str.contains(mode, na=False)] = None

    try:
        wait_play = pd.to_numeric(wait_play)
    except ValueError:
        print('没处理好数据啊！！！！！')
        raise ValueError
    wait_play.name = '0425s'
    print('finished')
    return wait_play
def clean_3189_f(feature):
    print('start 3189 fenlei')
    feature = feature.copy()
    wait_play = feature['3189']

    wait_play = wait_play.apply(strQ2B)
    wait_play = wait_play.apply(nums_ave)

    d = wait_play
    d = pd.to_numeric(d, errors='coerce')
    wait_play[~d.isna()] = np.nan

    wrong_index = ['\+\-', '3-5', '少数', '偶见', '未检出']
    for mode in wrong_index:
        wait_play[wait_play.str.contains(mode, na=False)] = np.nan

    wrong_index = ['\+']
    for mode in wrong_index:
        wait_play[wait_play.str.contains(mode, na=False)] = '阳性'

    wrong_index = ['\-']
    for mode in wrong_index:
        wait_play[wait_play.str.contains(mode, na=False)] = '阴性'

    symptom = [['阳性'],
               ['阴性', '正常', 'Normal', 'NormaL']]
    for cls, mode_list in enumerate(symptom):
        for mode in mode_list:
            wait_play[wait_play.str.contains(mode, na=False)] = '3189f_' + str(cls)

    wait_play = wait_play.fillna('3189f_' + str(len(symptom)))
    wait_play = pd.get_dummies(wait_play)
    wait_play = wait_play.drop(columns='3189f_' + str(len(symptom)))
    print('finished')
    return wait_play
def clean_3189_s(feature):
    print('start 3189 shuzhi')
    feature = feature.copy()
    wait_play = feature['3189']

    not_know_index = ['阴性', '未检出', '\+', '未见', '正常', '/LP', '/HP']  # 300005 3429
    for mode in not_know_index:
        wait_play[wait_play.str.contains(mode, na=False)] = np.nan

    not_know_index = ['少', '满', '脓白', '偶', 'Ⅱ']  # 300005 3429
    for mode in not_know_index:
        wait_play[wait_play.str.contains(mode, na=False)] = np.nan

    not_know_index = ['3.-4']  # 300005
    for mode in not_know_index:
        wait_play[wait_play.str.contains(mode, na=False)] = 3.5

    not_know_index = ['12.-15']  # 300005
    for mode in not_know_index:
        wait_play[wait_play.str.contains(mode, na=False)] = 13.5

    not_know_index = ['3.20.']  # 300005
    for mode in not_know_index:
        wait_play[wait_play.str.contains(mode, na=False)] = 3.2

    not_know_index = ['4--6']  # 3429
    for mode in not_know_index:
        wait_play[wait_play.str.contains(mode, na=False)] = 5

    not_know_index = ['>25']  # 3429
    for mode in not_know_index:
        wait_play[wait_play.str.contains(mode, na=False)] = 26

    wait_play = wait_play.apply(nums_ave)

    try:
        wait_play = pd.to_numeric(wait_play)
    except ValueError:
        print('没处理好数据啊！！！！！')
        raise ValueError
    wait_play.name = '3189s'
    print('finished')
    return wait_play
def clean_shell(feature):
    in_list = ['139', '143', '1474', '20002', '2177']
    hold_play = []
    for index_name in in_list:
        try:
            print('start ' + index_name)
            wait_play = feature[index_name]
            wait_play = wait_play.apply(strQ2B)
            wait_play = wait_play.apply(nums_ave)

            try:
                wrong_index = ['阴性', '\+']
                for mode in wrong_index:
                    wait_play[wait_play.str.contains(mode, na=False)] = np.nan

            except AttributeError:
                pass

            try:
                wait_play = pd.to_numeric(wait_play)
            except ValueError:
                print('没处理好数据啊！！！！！')
                raise ValueError
            print('finished')

            hold_play.append(wait_play)

        except KeyError:
            print(index_name + ' no exist!')

    return hold_play
def clean_shell2(feature):
    in_list = ['2228', '2229', '2230', '2231', '2233', '2278', '2282', '3433', '3485', '3486']
    hold_play = []
    for index_name in in_list:
        try:
            print('start ' + index_name)
            wait_play = feature[index_name]
            d = wait_play
            d = pd.to_numeric(d, errors='coerce')
            wait_play[~d.isna()] = None

            label_index = ['极弱阳', '可疑', '\+\-', '<500', '少见', '未检出', '查见', '检出', '检到', '结果见TCT']
            for mode in label_index:
                wait_play[wait_play.str.contains(mode, na=False)] = None

            label_index = ['\+', '重度', '阳性']
            for mode in label_index:
                wait_play[wait_play.str.contains(mode, na=False)] = '阳性'

            label_index = ['\-', '阴性']
            for mode in label_index:
                wait_play[wait_play.str.contains(mode, na=False)] = '阴性'

            symptom = [['阴性'],
                       ['阳性']]
            for cls, mode_list in enumerate(symptom):
                for mode in mode_list:
                    wait_play[wait_play.str.contains(mode, na=False)] = '{}_'.format(index_name) + str(cls)

            wait_play = wait_play.fillna('{}_'.format(index_name) + str(len(symptom)))

            wait_play = pd.get_dummies(wait_play)
            wait_play = wait_play.drop(columns='{}_'.format(index_name) + str(len(symptom)))
            print('finished')

            hold_play.append(wait_play)

        except KeyError:
            print(index_name + ' no exist!')

    return hold_play
def clean_shell3(feature):
    in_list = ['300005', '3429']
    hold_play = []
    for index_name in in_list:
        try:
            print('start ' + index_name)
            wait_play = feature[index_name]

            not_know_index = ['阴性', '未检出', '\+', '未见', '正常', '/LP', '/HP']  # 300005 3429
            for mode in not_know_index:
                wait_play[wait_play.str.contains(mode, na=False)] = np.nan

            not_know_index = ['少', '满', '脓白', '偶', 'Ⅱ']  # 300005 3429
            for mode in not_know_index:
                wait_play[wait_play.str.contains(mode, na=False)] = np.nan

            not_know_index = ['3.-4']  # 300005
            for mode in not_know_index:
                wait_play[wait_play.str.contains(mode, na=False)] = 3.5

            not_know_index = ['12.-15']  # 300005
            for mode in not_know_index:
                wait_play[wait_play.str.contains(mode, na=False)] = 13.5

            not_know_index = ['3.20.']  # 300005
            for mode in not_know_index:
                wait_play[wait_play.str.contains(mode, na=False)] = 3.2

            not_know_index = ['4--6']  # 3429
            for mode in not_know_index:
                wait_play[wait_play.str.contains(mode, na=False)] = 5

            not_know_index = ['>25']  # 3429
            for mode in not_know_index:
                wait_play[wait_play.str.contains(mode, na=False)] = 26

            wait_play = wait_play.apply(nums_ave)

            try:
                wait_play = pd.to_numeric(wait_play)
            except ValueError:
                print('没处理好数据啊！！！！！')
                raise ValueError
            print('finished')

            hold_play.append(wait_play)

        except KeyError:
            print(index_name + ' no exist!')

    return hold_play

def clean_words_(feature):
    in_list = ['0101', '0102', '0113', '0114', '0115', '0116', '0117', '0118', '0119', '0120',
               '0121', '0122', '0123', '0124', '0201', '0203', '0208', '0209', '0210', '0215',
               '0217', '0222', '0225', '0405', '0409', '0413', '0422', '0426', '0429',
               '0434', '0435', '0501', '0503', '0509', '0516', '0537', '0539', '0541', '0703',
               '0705', '0706', '0707', '0709', '0726', '0731', '0911', '0912', '0929', '0947',
               '0949', '0954', '0972', '0978', '0984', '100010', '1001', '1102', '1103', '1302',
               '1305', '1314', '1315', '1316', '1329', '1330', '1402', '2501', '3301', '3430',
               '4001', 'A201', 'A202', 'A301', 'A302']
    hold_play = []
    for index_name in in_list:
        # Assign the storage path of model
        temp_dict_path = '../data/temp_dict/'
        dict_pattern = temp_dict_path + 'dict_{}.txt'.format(index_name)
        text_pattern = temp_dict_path + 'text_{}.txt'.format(index_name)
        model_pattern = temp_dict_path + 'model_{}.doc2vec'.format(index_name)
        if not os.path.exists(temp_dict_path):
            os.mkdir(temp_dict_path)

        try:
            print('start ' + index_name)

            # Read in data, and change its mode
            wait_play_original = feature[index_name]
            wait_play = wait_play_original.dropna()
            word_list = list(wait_play)
            word_join = ' '.join(word_list)

            # 获取词频
            if os.path.exists(dict_pattern):
                jieba.load_userdict(dict_pattern)
                print(' [*] The dict file exist, load successfully!')
                pass
            else:
                print(' [*] The dict file does not exist, try to create!')
                try:
                    jieba.enable_parallel(200)
                    word_split = [x for x in jieba.cut(word_join) if len(x) >= 2]
                    jieba.disable_parallel()
                except NotImplementedError:
                    word_split = [x for x in jieba.cut(word_join) if len(x) >= 2]
                word_counter = Counter(word_split)
                len_word_counter = len(word_counter)
                word_freq = word_counter.most_common(int(len_word_counter * 0.9) + 1)
                print(' [*] The dict file is created, with length of {}'.format(len(word_freq)))
                with codecs.open(dict_pattern, 'w+', 'UTF-8') as f:
                    for x in word_freq:
                        f.write('{0}\t{1}\n'.format(x[0], x[1]))
                f.close()
                print(' [*] The dict file is create!')
                jieba.load_userdict(dict_pattern)

            def split(ss):
                in_str = str(ss)
                if in_str == 'nan':
                    return ss
                else:
                    return ' '.join(jieba.cut(ss))

            # 分词
            wait_play = wait_play.apply(split)
            if os.path.exists(text_pattern):
                print(' [*] The text file exist!')
                pass
            else:
                print(' [*] The text file does not exist, try to create!')
                wait_play.to_csv(text_pattern, sep='\n', index=False)
                print(' [*] The text file is create!')

            # doc2vec parameters
            vector_size = 5  # 300维
            window_size = 5
            min_count = 15
            sampling_threshold = 0
            negative_size = 0
            train_epoch = 20
            dm = 0  # 0 = dbow; 1 = dmpv
            worker_count = 200  # number of parallel processes

            if not os.path.exists(model_pattern):
                print(' [*] The model file does not exist, try to create!')
                docs = gensim.models.doc2vec.TaggedLineDocument(text_pattern)
                model_doc = Doc2Vec(docs, vector_size=vector_size, window=window_size,
                                    min_count=min_count, sample=sampling_threshold,
                                    workers=worker_count, hs=0, dm=dm,
                                    negative=negative_size, dbow_words=1, dm_concat=1, epochs=train_epoch)
                model_doc.save(model_pattern)
                print(' [*] The model file is create!')
            else:
                print(' [*] The model file exist!')
                model_doc = Doc2Vec.load(model_pattern)
                vector_size = model_doc.vector_size

            def str2vec(ss):
                list_ss = list(model_doc.infer_vector(ss.split(sep=' ')))
                list_ss_str = [str(num) for num in list_ss]
                return '/'.join(list_ss_str)

            wait_play = wait_play.apply(str2vec)
            wait_play_original[~wait_play_original.isna()] = wait_play

            out_ones = wait_play_original.str.split('/', expand=True)
            new_index = []
            for i in range(vector_size):
                new_index.append('{}_{}'.format(index_name, i))
            out_ones.columns = new_index
            hold_play.append(out_ones)
        except KeyError:
            print(index_name + ' no exist!')

    return hold_play

def test_series(feature):
    index_format = '31%02d'
    list_min = 0
    list_max = 99
    hold_play = {}
    for index in range(list_min, list_max+1):
        index_name = index_format % index
        try:
            print('start ' + index_name)
            wait_play = feature[index_name]
            wait_play = wait_play.apply(strQ2B)
            wait_play = wait_play.apply(nums_ave)

            try:
                wrong_index = []  # 979001 979002 979003
                for mode in wrong_index:
                    wait_play[wait_play.str.contains(mode, na=False)] = np.nan

            except AttributeError:
                pass

            try:
                wait_play = pd.to_numeric(wait_play, errors='ignore')
            except ValueError:
                print('没处理好数据啊！！！！！')
                raise ValueError
            print('finished')

            hold_play[index] = wait_play.count()

        except KeyError:
            print(index_name + ' no exist!')
    print(hold_play)
    return hold_play


def feature_process():
    data_path = '../data/tmp_feature_raw.csv'
    data_path_no_clean = '../data/tmp_feature_select.csv'
    feature = pd.read_csv(data_path)
    out_list = pd.read_csv(data_path_no_clean)

    # out_list = my_concat(out_list, test_series(feature))

    out_list = my_concat(out_list, clean_11__(feature))
    out_list = my_concat(out_list, clean_1000_(feature))
    out_list = my_concat(out_list, clean_18__(feature))
    out_list = my_concat(out_list, clean_2302(feature))
    out_list = my_concat(out_list, clean_23__(feature))
    out_list = my_concat(out_list, clean_2690__(feature))
    out_list = my_concat(out_list, clean_3__(feature))
    out_list = my_concat(out_list, clean_3000__(feature))
    out_list = my_concat(out_list, clean_31__(feature))
    out_list = my_concat(out_list, clean_31__2(feature))

    out_list = my_concat(out_list, clean_6690__(feature))
    out_list = my_concat(out_list, clean_8090__(feature))
    out_list = my_concat(out_list, clean_9790__(feature))

    # old
    out_list = my_concat(out_list, clean_0424(feature))
    out_list = my_concat(out_list, clean_30007(feature))
    out_list = my_concat(out_list, clean_100006(feature))
    out_list = my_concat(out_list, clean_2174(feature))
    out_list = my_concat(out_list, clean_183(feature))
    out_list = my_concat(out_list, clean_190(feature))
    out_list = my_concat(out_list, clean_191(feature))
    out_list = my_concat(out_list, clean_192(feature))
    out_list = my_concat(out_list, clean_193(feature))

    out_list = my_concat(out_list, clean_31(feature))
    out_list = my_concat(out_list, clean_314(feature))
    out_list = my_concat(out_list, clean_32(feature))
    out_list = my_concat(out_list, clean_320(feature))
    out_list = my_concat(out_list, clean_38(feature))
    out_list = my_concat(out_list, clean_312(feature))
    out_list = my_concat(out_list, clean_313(feature))
    out_list = my_concat(out_list, clean_155(feature))
    out_list = my_concat(out_list, clean_1321(feature))
    out_list = my_concat(out_list, clean_1322(feature))

    out_list = my_concat(out_list, clean_0202(feature))
    out_list = my_concat(out_list, clean_0206(feature))
    out_list = my_concat(out_list, clean_0207(feature))
    out_list = my_concat(out_list, clean_0212(feature))
    out_list = my_concat(out_list, clean_0213(feature))
    out_list = my_concat(out_list, clean_0216(feature))
    out_list = my_concat(out_list, clean_3399(feature))
    out_list = my_concat(out_list, clean_3400(feature))
    out_list = my_concat(out_list, clean_3601(feature))
    out_list = my_concat(out_list, clean_0901(feature))

    out_list = my_concat(out_list, clean_0981(feature))
    out_list = my_concat(out_list, clean_0983(feature))
    out_list = my_concat(out_list, clean_0406(feature))
    out_list = my_concat(out_list, clean_0407(feature))
    out_list = my_concat(out_list, clean_0415(feature))
    out_list = my_concat(out_list, clean_0420(feature))
    out_list = my_concat(out_list, clean_0423(feature))
    out_list = my_concat(out_list, clean_0427(feature))
    out_list = my_concat(out_list, clean_0430(feature))
    out_list = my_concat(out_list, clean_0431(feature))
    out_list = my_concat(out_list, clean_0434(feature))
    out_list = my_concat(out_list, clean_0439(feature))

    out_list = my_concat(out_list, clean_0440(feature))
    out_list = my_concat(out_list, clean_0728(feature))
    out_list = my_concat(out_list, clean_0732(feature))
    out_list = my_concat(out_list, clean_1328(feature))
    out_list = my_concat(out_list, clean_100014(feature))
    out_list = my_concat(out_list, clean_1325(feature))
    out_list = my_concat(out_list, clean_1326(feature))
    out_list = my_concat(out_list, clean_2409(feature))
    out_list = my_concat(out_list, clean_0421(feature))

    out_list = my_concat(out_list, clean_BMI(feature))

    out_list = my_concat(out_list, clean_0425_f(feature))
    out_list = my_concat(out_list, clean_0425_s(feature))
    out_list = my_concat(out_list, clean_3189_f(feature))
    out_list = my_concat(out_list, clean_3189_s(feature))
    out_list = my_concat(out_list, clean_shell(feature))
    out_list = my_concat(out_list, clean_shell2(feature))
    out_list = my_concat(out_list, clean_shell3(feature))

    out_list = my_concat(out_list, clean_words_(feature))
    out_list.to_csv('../data/tmp_feature_final.csv', index=False)


if __name__ == '__main__':
    feature_process()
