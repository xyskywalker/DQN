# coding:utf-8
import numpy as np
import pandas as pd
import datetime


#print(datetime.datetime.now(), 'Start - Load Data')
#csv_train = pd.read_csv('df_test.csv').fillna(value=0.0)
#print(datetime.datetime.now(), 'End - Load Data')
#csv_train.drop(['交易时间','申报受理时间', '住院开始时间','住院终止时间','操作时间','出院诊断病种名称'], axis=1, inplace=True)

#np.save('arr_test.npy', csv_train)
print(datetime.datetime.now(), 'Start - Load Data')
arr_train = np.load('arr_test.npy')
#train_data = np.load('/media/xy/247E930D7E92D740/ShareData/train_data.npy')
print(datetime.datetime.now(), 'End - Load Data')

#df_test = pd.DataFrame(arr_train)

#print(df_test.groupby(by=1)[1].count().max()) #Test Data: 907

#print(train_data.shape)
#print(train_data[0].shape)
#print(train_data[0])

#print(arr_train)
#print(arr_train[0])
#print(arr_train[0][0])
#print(arr_train[0][1])
#print(arr_train[0][2])
#print(arr_train[0][3])
#print(datetime.datetime.now(), 'Start - Load Data')
df_train = pd.DataFrame(arr_train)
df_label = pd.read_csv('df_id_test.csv',header=-1, encoding='utf-8')
arr_label = np.array(df_label)

train_list = list(range(len(arr_label)))

#print(np.array(df_train[df_train[1]==352121004173837].drop([0, 1], axis=1)).shape)
arr_len = np.zeros([4000], dtype=np.int32)
for i in range(4000):
    item_arr = np.zeros([1415, 61])
    #print(df_train[df_train[1]==arr_label[i][0]].sort_values(by=0, ascending=True))
    #print(df_train[df_train[1]==arr_label[i][0]].sort_values(by=0, ascending=True).drop([0, 1], axis=1))
    item_temp = np.array(df_train[df_train[1]==arr_label[i][0]].sort_values(by=0, ascending=True).drop([0, 1], axis=1))
    item_arr[0:len(item_temp), ] = item_temp
    arr_len[i] = len(item_temp)
    #print(item_temp)

    train_list[i] = item_arr

    if i % 10 == 0:
        print('Steps:', i)

#print(train_list)
#np.save('/media/xy/247E930D7E92D740/ShareData/test_data.npy', train_list)
np.save('test_len.npy', arr_len)

#print(datetime.datetime.now(), 'End - Load Data')
#print(df_train[df_train[1] == 352120001523108])

#print(csv_train.groupby(by='出院诊断病种名称')['出院诊断病种名称'].count())


#print(csv_train.groupby(by='个人编码')['个人编码'].count().max())  #1415

#fee_detail = pd.read_csv('fee_detail.csv')

#print(fee_detail.head())

