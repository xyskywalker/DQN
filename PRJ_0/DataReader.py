# coding:utf-8
import numpy as np
import pandas as pd
import datetime
import functools

#print(datetime.datetime.now(), 'Start - Load Data')
#csv_train = pd.read_csv('df_train.csv').fillna(value=0.0)
#print(datetime.datetime.now(), 'End - Load Data')
#csv_train.drop(['交易时间','申报受理时间', '住院开始时间','住院终止时间','操作时间','出院诊断病种名称'], axis=1, inplace=True)

#np.save('arr_train.npy', csv_train)
print(datetime.datetime.now(), 'Start - Load Data')
arr_train = np.load('arr_train.npy')
print(datetime.datetime.now(), 'End - Load Data')

print(arr_train)
print(arr_train[0])
print(arr_train[0][0])
print(arr_train[0][1])
print(arr_train[0][2])
print(arr_train[0][3])
print(datetime.datetime.now(), 'Start - Load Data')
df_train = pd.DataFrame(arr_train)
print(datetime.datetime.now(), 'End - Load Data')
print(df_train[df_train[1] == 352120001523108])

#print(csv_train.groupby(by='出院诊断病种名称')['出院诊断病种名称'].count())


#print(csv_train.groupby(by='个人编码')['个人编码'].count().max())  #1415

#fee_detail = pd.read_csv('fee_detail.csv')

#print(fee_detail.head())

