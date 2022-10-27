'''
Author: RenzeLou marionojump0722@gmail.com
Date: 2022-07-13 23:02:09
LastEditors: RenzeLou marionojump0722@gmail.com
LastEditTime: 2022-10-26 14:08:53
FilePath: /bare-RUN/test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# from multiprocessing import pool
# from pyexpat import model
# from transformers import RobertaTokenizer,RobertaModel,BertTokenizer,BertModel, BertConfig
# import torch
# from traceback import print_tb
# import scipy.sparse as sp

# model_ckpt='roberta-base'
# model=RobertaModel.from_pretrained(model_ckpt)
# tokenizer=RobertaTokenizer.from_pretrained(model_ckpt)
# text1='<s> hello,have a good day </s></s> eat an apple </s></s> see you agagin </s>'
# inputs=tokenizer.tokenize(text1)
# print(inputs)



# config = BertConfig.from_pretrained(model_ckpt)

from cgi import print_arguments
# from curses import window
import torch
import pickle

import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import json

know_path = '/home/DATA1/gxj/bare-RUN/dd_data/knowledge_train.pkl'
processed_knowledge_path = '/home/DATA1/gxj/bare-RUN/dd_data/base_knowledge_processed_train.pkl'
data_path = '/home/DATA1/gxj/bare-RUN/dd_data/dailydialog_' + 'train' + '.pkl'
processed_data_path = '/home/DATA1/gxj/bare-RUN/dd_data/dailydialog_train_processed.pkl'
base_know_processed_path = '/home/DATA1/gxj/bare-RUN/dd_data/base_knowledge_processed_train.pkl'
window_base_know_processed_path = '/home/DATA1/gxj/bare-RUN/dd_data/window_base_knowledge_processed_'+'train'+'.pkl'
base_know_path = '/home/DATA1/gxj/bare-RUN/dd_data/base_knowledge_'+'train'+'.pkl'
base_know_path = '/home/data1/gxj/1-bare-RUN/dd_data/base_knowledge_'+'train'+'.pkl'
path = '/home/DATA1/gxj/7-bare-RUN/dd_data/dailydialog_train.pkl'


# know = pickle.load(open(processed_knowledge_path, 'rb'), encoding='latin1')
data = pickle.load(open(path, 'rb'), encoding='latin1')
print(data[1])
# # base_know = pickle.load(open(base_know_path, 'rb'), encoding='latin1')
# print(data[0][0])
# base_know_path = '/home/data1/gxj/2-bare-RUN/dd_data/base_knowledge_'+'test'+'.pkl'
# iemocap_path = '/home/data1/gxj/iemocap_test.json'
# dd_path = '/home/data1/gxj/dailydialog_train.json'

# f = open(iemocap_path,'r',encoding='utf-8')
# m = json.load(f)
# for i in m:
#     print(m[i][0])
#     break

num_relation = 0
window = 2
# anger sadness fear

# DD = [9, 7, 4, 14, 8, 21, 18, 10, 16, 10, 10, 12, 12, 12, 14, 8, 10, 11, 11, 10, 17, 12, 6, 5, 10, 10, 16, 6, 12, 17, 10, 10, 4, 8, 9, 13, 12, 10, 11, 11, 19, 12, 4, 12, 15, 10, 8, 10, 6, 10, 8, 9, 16, 21, 13, 9, 8, 4, 10, 8, 10, 12, 10, 6, 8, 5, 6, 12, 4, 14, 7, 8, 8, 12, 8, 13, 14, 16, 15, 13, 11, 16, 17, 14, 19, 12, 23, 10, 6, 10, 10, 17, 11, 6, 14, 12, 7, 10, 8, 12, 7, 7, 13, 8, 14, 4, 12, 14, 10, 16, 6, 13, 16, 10, 14, 12, 12, 6, 6, 16, 9, 14, 13, 12, 16, 23, 11, 4, 13, 10, 8, 21, 12, 10, 5, 12, 11, 6, 6, 14, 11, 7, 8, 11, 10, 6, 14, 8, 13, 11, 10, 15, 12, 14, 6, 12, 14, 8, 9, 15, 9, 15, 13, 11, 12, 16, 12, 7, 6, 7, 12, 4, 13, 19, 5, 10, 4, 18, 11, 12, 11, 14, 7, 10, 12, 4, 13, 4, 11, 10, 7, 12, 4, 7, 12, 5, 8, 7, 15, 4, 6, 19, 12, 8, 13, 16, 8, 7, 14, 5, 10, 11, 12, 14, 8, 4, 8, 17, 11, 12, 12, 7, 6, 8, 10]
# IE = [26, 37, 26, 53, 54, 26, 54, 35, 43, 28, 26, 26, 110, 42, 28, 51]
# len = len(IE)
# print(3/6.804444444444444)
# for i in IE:
#     if (i-1)>2*(window-1):
#         num_relation += window+2
#     else:
#         num_relation += round((i-1)/2)+2
# print(num_relation)
# print(num_relation/len)
# test_fscore_list = []
# test_pos_list = []
# test_neg_list = []

# iemocap_fscore_list = [0.8586,0.8514,0.8527,0.8467,0.8474]
# iemocap_pos_list = []
# iemocap_neg_list = []

# test_fscore_mean = np.round(np.mean(test_fscore_list) * 100, 2)
# test_fscore_std = np.round(np.std(test_fscore_list) * 100, 2)
# test_pos_mean = np.round(np.mean(test_pos_list) * 100, 2)
# test_pos_std = np.round(np.std(test_pos_list) * 100, 2)
# test_neg_mean = np.round(np.mean(test_neg_list) * 100, 2)
# test_neg_std = np.round(np.std(test_neg_list) * 100, 2)

# iemocap_fscore_mean = np.round(np.mean(iemocap_fscore_list) * 100, 2)
# iemocap_fscore_std = np.round(np.std(iemocap_fscore_list) * 100, 2)
# iemocap_pos_mean = np.round(np.mean(iemocap_pos_list) * 100, 2)
# iemocap_pos_std = np.round(np.std(iemocap_pos_list) * 100, 2)
# iemocap_neg_mean = np.round(np.mean(iemocap_neg_list) * 100, 2)
# iemocap_neg_std = np.round(np.std(iemocap_neg_list) * 100, 2)

# log_lines = f'test fscore: {test_fscore_mean}(+-{test_fscore_std})'
# print(log_lines)
# log_lines = f'test pos fscore: {test_pos_mean}(+-{test_pos_std})'
# print(log_lines)
# log_lines = f'test neg fscore: {test_neg_mean}(+-{test_neg_std})'
# print(log_lines)
# print()
# log_lines = f'iemocap fscore: {iemocap_fscore_mean}(+-{iemocap_fscore_std})'
# print(log_lines)
# log_lines = f'iemocap pos fscore: {iemocap_pos_mean}(+-{iemocap_pos_std})'
# print(log_lines)
# log_lines = f'iemocap neg fscore: {iemocap_neg_mean}(+-{iemocap_neg_std})'
# print(log_lines)

# DD_y = [84.12,84.29,85.14,84.58,84.3,84.5,84.46,84.18]
# print(len(DD_y))