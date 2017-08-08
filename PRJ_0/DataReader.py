# coding:utf-8
import numpy as np
import pandas as pd
import datetime
import tensorflow as tf
import tensorflow.contrib as tfc
'''
#len_test = np.load('test_len.npy')
#print(len_test)
#print(datetime.datetime.now(), 'Start - Load Data')
#csv_train = pd.read_csv('df_test.csv').fillna(value=0.0)
#print(datetime.datetime.now(), 'End - Load Data')
#csv_train.drop(['交易时间','申报受理时间', '住院开始时间','住院终止时间','操作时间','出院诊断病种名称'], axis=1, inplace=True)

#np.save('arr_test.npy', csv_train)
print(datetime.datetime.now(), 'Start - Load Data')
arr_train = np.load('arr_train.npy')
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
df_label = pd.read_csv('df_id_train.csv',header=-1, encoding='utf-8')
arr_label = np.array(df_label)

train_list = list(range(len(arr_label)))

#print(np.array(df_train[df_train[1]==352121004173837].drop([0, 1], axis=1)).shape)

arr_len = np.zeros([20000], dtype=np.int32)
for i in range(20000):
    item_arr = np.zeros([1415, 61])
    #print(df_train[df_train[1]==arr_label[i][0]].sort_values(by=0, ascending=True))
    #print(df_train[df_train[1]==arr_label[i][0]].sort_values(by=0, ascending=True).drop([0, 1], axis=1))
    #item_temp = np.array(df_train[df_train[1]==arr_label[i][0]].sort_values(by=0, ascending=True).drop([0, 1], axis=1))
    #item_arr[0:len(item_temp), ] = item_temp
    item_temp = np.array(df_train[df_train[1] == arr_label[i][0]])
    arr_len[i] = len(item_temp)
    #print(item_temp)

    train_list[i] = item_arr

    if i % 10 == 0:
        print('Steps:', i)

#print(train_list)
#np.save('/media/xy/247E930D7E92D740/ShareData/test_data.npy', train_list)
np.save('train_len.npy', arr_len)

arr_len = np.array(df_train.groupby(by=1)[1].count())
print(arr_len)
np.save('train_len.npy', arr_len)
'''
#print(datetime.datetime.now(), 'End - Load Data')
#print(df_train[df_train[1] == 352120001523108])

#print(csv_train.groupby(by='出院诊断病种名称')['出院诊断病种名称'].count())


#print(csv_train.groupby(by='个人编码')['个人编码'].count().max())  #1415

#fee_detail = pd.read_csv('fee_detail.csv')

#print(fee_detail.head())


def print_activations(t):
    print(t.op.name, '' , t.get_shape().as_list())

env_len = 1415
env_d = 61
learning_rate = 0.0001

envInput = tf.placeholder(shape=[None, env_len, env_d], dtype=tf.float32)
envIn = tf.reshape(envInput, shape=[-1, env_len, env_d, 1])

conv1 = tfc.layers.convolution2d(inputs=envIn,
                                 num_outputs=512,
                                 kernel_size=[3, 3],
                                 stride=[3, 1],
                                 padding='VALID',
                                 biases_initializer=None)
pool1 = tfc.layers.max_pool2d(inputs=conv1, kernel_size=[2, 2], stride=[1, 1], padding='VALID')

conv2 = tfc.layers.convolution2d(inputs=pool1,
                                 num_outputs=512,
                                 kernel_size=[3, 3],
                                 stride=[3, 1],
                                 padding='VALID',
                                 biases_initializer=None)
pool2 = tfc.layers.max_pool2d(inputs=conv2, kernel_size=[2, 2], stride=[1, 1], padding='VALID')

conv3 = tfc.layers.convolution2d(inputs=pool2,
                                 num_outputs=256,
                                 kernel_size=[3, 3],
                                 stride=[3, 1],
                                 padding='VALID',
                                 biases_initializer=None)
pool3 = tfc.layers.max_pool2d(inputs=conv3, kernel_size=[2, 2], stride=[1, 1], padding='VALID')

conv4 = tfc.layers.convolution2d(inputs=pool3,
                                 num_outputs=256,
                                 kernel_size=[3, 3],
                                 stride=[2, 2],
                                 padding='VALID',
                                 biases_initializer=None)
pool4 = tfc.layers.max_pool2d(inputs=conv4, kernel_size=[2, 2], stride=[1, 1], padding='VALID')

conv5 = tfc.layers.convolution2d(inputs=pool4,
                                 num_outputs=128,
                                 kernel_size=[3, 3],
                                 stride=[2, 2],
                                 padding='VALID',
                                 biases_initializer=None)
pool5 = tfc.layers.max_pool2d(inputs=conv5, kernel_size=[2, 2], stride=[1, 1], padding='VALID')

conv6 = tfc.layers.convolution2d(inputs=pool5,
                                 num_outputs=64,
                                 kernel_size=[3, 3],
                                 stride=[2, 2],
                                 padding='VALID',
                                 biases_initializer=None)
pool6 = tfc.layers.max_pool2d(inputs=conv6, kernel_size=[2, 2], stride=[1, 1], padding='VALID')


print_activations(conv1)
print_activations(conv2)
print_activations(conv3)
print_activations(conv4)
print_activations(conv5)
print_activations(conv6)
print_activations(pool1)
print_activations(pool2)
print_activations(pool3)
print_activations(pool4)
print_activations(pool5)
print_activations(pool6)