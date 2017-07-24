# coding:utf-8
import numpy as np
import pandas as pd
import datetime
import os

#lines = open('gy_contest_link_traveltime_training_data.txt').readlines()
#fp = open('training_data.txt','w')
#for s in lines:
#    # replace是替换，write是写入
#    fp.write( s.replace(';',',').replace('[','').replace(')','')
#              .replace('time_interval', 'time_interval-s,time_interval-e'))
#fp.close()  # 关闭文件

#df_train = pd.read_csv('training_data.txt').sort_values(by=['link_ID', 'time_interval-s'])
#print(df_train.head(100))

#df_train[df_train['link_ID'] == 3377906287934510514].to_csv('3377906287934510514.csv')

#df_train[df_train['link_ID'] == 9377906282776510514].to_csv('9377906282776510514.csv')

df_link = pd.read_csv('gy_contest_link_info.txt', sep=';')
print(df_link)