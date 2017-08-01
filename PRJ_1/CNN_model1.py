import tensorflow as tf
import tensorflow.contrib as tfc
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random
import math
import pandas as pd

train_data_all = np.load('train_data.npy')
train_data_arr = []
train_data_mean = []
train_data_stddev = []

test_data_arr = []
test_data_mean = []
test_data_stddev = []
for train_data in train_data_all:
    df_train_data = pd.DataFrame(train_data).sort_values(by=[7, 0]).copy()
    df_test_data = pd.DataFrame(train_data).sort_values(by=[7, 0]).copy()

    df_train_data = df_train_data[df_train_data[2] < 6.0]
    df_test_data = df_test_data[df_test_data[2] >= 6.0]

    ##########Train Data##########
    train_data_ = np.array(df_train_data)
    train_data_[:,8] = train_data_[:,9] / train_data_[:,8] # 通过时间折算为速度
    min_speed = min(train_data_[:, 8])
    max_speed = max(train_data_[:, 8])
    train_data_[:, 8] -= min_speed
    train_data_[:, 8] /= max_speed
    #mean = np.average(train_data_, axis=0) + 0.00001
    #stddev = np.std(train_data_, axis=0) + 0.00001

    #train_data_ = train_data_ - mean
    #train_data_ = train_data_ / stddev

    train_data_arr.append(train_data_)
    #train_data_mean.append(mean)
    #train_data_stddev.append(stddev)

    ##########Test Data##########
    test_data_ = np.array(df_test_data)
    test_data_[:,8] = test_data_[:,9] / test_data_[:,8] # 通过时间折算为速度
    #test_mean = np.average(test_data_, axis=0) + 0.00001
    #test_stddev = np.std(test_data_, axis=0) + 0.00001

    #test_data_ = test_data_ - test_mean
    #test_data_ = test_data_ / test_stddev

    test_data_arr.append(test_data_)
    #test_data_mean.append(test_mean)
    #test_data_stddev.append(test_stddev)

print('train_data', len(train_data_arr))
print('train_data.shape', train_data_arr[0].shape)

#train_data = train_data_arr[15]
#line1 = train_data[92 * 6 : 92 * 6 + 720, 8]
#plt.plot(line1)
#line1 = train_data[720 * 7 : 720 * 7 + 720, 8]
#plt.plot(line1)
#line1 = train_data[720 * 8 : 720 * 8 + 720, 8]
#plt.plot(line1)
#line1 = train_data[720 * 9 : 720 * 9 + 720, 8]
#plt.plot(line1)
#plt.show()


def generate_data(isTrain, batch_size, start_link, start_day, start_time_piece):
    x, y = [], []
    for batch in range(batch_size):
        x_ = np.random.uniform(0, 1, [60, 60])
        y_ = np.random.uniform(0, 1, [60])
        time_piece = start_time_piece + batch
        if time_piece >= 720:
            time_piece -= 720
        for i in range(60):
            i_link = start_link + i
            if i_link >= 132:
                i_link -= 132
            train_data = train_data_arr[i_link]
            start_ = time_piece * 92 + start_day
            end_ = start_ + 60
            row = train_data[start_: end_, 8]
            x_[i] = row
            y_[i] = train_data[end_][8]
        x.append(x_)
        y.append(y_)

    return x, y

def print_activations(t):
    print(t.op.name, '' , t.get_shape().as_list())

learning_rate = 0.001

X = tf.placeholder(shape=[None, 60, 60], dtype=tf.float32)
X_ = tf.reshape(X, shape=[-1, 60, 60, 1])
Y = tf.placeholder(shape=[None, 60], dtype=tf.float32)

conv1 = tfc.layers.convolution2d(inputs=X_,
                                 num_outputs=2048,
                                 kernel_size=[3, 3],
                                 stride=[2, 2],
                                 padding='VALID',
                                 biases_initializer=None)
conv2 = tfc.layers.convolution2d(inputs=conv1,
                                 num_outputs=1024,
                                 kernel_size=[2, 2],
                                 stride=[1, 1],
                                 padding='VALID',
                                 biases_initializer=None)
conv3 = tfc.layers.convolution2d(inputs=conv2,
                                 num_outputs=512,
                                 kernel_size=[2, 2],
                                 stride=[1, 1],
                                 padding='VALID',
                                 biases_initializer=None)

pool1 = tfc.layers.max_pool2d(inputs=conv3, kernel_size=[2, 2], stride=[1, 1], padding='VALID')


# 全连接层
# 权重
W_fc1 = tf.get_variable('W_fc1', shape=[26 * 26 * 512, 1024], initializer=tf.contrib.layers.xavier_initializer())
# 偏置
b_fc1 = tf.get_variable('b_fc1', shape=[1024], initializer=tf.contrib.layers.xavier_initializer())
# 将池化输出转换为一维
h_pool1_flat = tf.reshape(pool1, [-1, 26 * 26 * 512])
# 激活函数
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

# Dropout层，避免过拟合
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

# 输出层
w_out = tf.get_variable('w_actiontype', shape=[1024, 60], initializer=tf.contrib.layers.xavier_initializer())
b_out = tf.get_variable('b_actiontype', shape=[60], initializer=tf.contrib.layers.xavier_initializer())
output = tf.matmul(h_fc1_drop, w_out) + b_out

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    x, y = generate_data(True, 60, 10, 10, 10)

    p = sess.run(output, feed_dict={X:x})
    print_activations(pool1)
    print(np.array(p))
    print(np.array(p).shape)

