import numpy as np
import pandas as pd
import datetime
import random
import tensorflow as tf
import tensorflow.contrib as tfc

print(datetime.datetime.now(), 'Start - Load Data')
train_data = np.load('/media/xy/247E930D7E92D740/ShareData/train_data.npy')
#train_data = np.load('train_data.npy')
df_label = pd.read_csv('df_id_train.csv',header=-1, encoding='utf-8')
arr_label = np.array(df_label)
id_list_1 = np.array(df_label[df_label[1] == 1].index)
id_list_0 = np.array(df_label[df_label[1] == 0].index)

train_len = np.load('train_len.npy')

print(datetime.datetime.now(), 'End - Load Data')

#print(train_data.shape) #(20000, 1415, 61)
#print(train_data[0].shape) #(1415, 61)


def get_data(batch_size = 50, is_train = True):
    # 1, 0的数据各随机获取 batch_size/2
    half_batch = int(batch_size/2)
    if is_train:
        list_1 = list(id_list_1[:950])
        list_0 = list(id_list_0[:18000])
    else:
        list_1 = list(id_list_1[950:])
        list_0 = list(id_list_0[18000:])
    id_1 = random.sample(list_1, half_batch)
    id_0 = random.sample(list_0, half_batch)
    train_x = np.zeros([50, 1415, 61], dtype=np.float32)
    train_y = np.zeros([50], dtype=np.int32)
    # 随机取 batch_size/2 的ID，用来随机交错填入1和0的数据
    random_ids = random.sample(range(batch_size), half_batch)
    i_1 = 0
    i_0 = 0
    for i in range(batch_size):
        if i in random_ids:
            train_x[i] = pre_data(id_1[i_1])
            train_y[i] = 1
            i_1 += 1
        else:
            train_x[i] = pre_data(id_0[i_0])
            train_y[i] = 0
            i_0 += 1

    return train_x, train_y


def pre_data(index_id):
    data = train_data[index_id]
    data_len = train_len[index_id]
    data_new = np.zeros(data.shape, dtype=data.dtype)
    start_index = random.randint(0, 1415 - data_len)
    data_new[start_index:start_index+data_len,] = data[0:data_len]
    data_min = np.min(data_new, axis=0)
    data_max = np.max(data_new, axis=0) + 0.000001
    data_new -= data_min
    data_new /= data_max
    return data_new

x, y = get_data()

def get_check_data():
    x = train_data[19000:]
    y = arr_label[19000:,1]
    return x, y

#print(x)
#print(y)

#print(pre_data(5))

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
                                 stride=[1, 1],
                                 padding='VALID',
                                 biases_initializer=None)
pool4 = tfc.layers.max_pool2d(inputs=conv4, kernel_size=[2, 2], stride=[1, 1], padding='VALID')

# 全连接层
# 权重
W_fc1 = tf.get_variable('W_fc1', shape=[47*49*256, 1024], initializer=tf.contrib.layers.xavier_initializer())
# 偏置
b_fc1 = tf.get_variable('b_fc1', shape=[1024], initializer=tf.contrib.layers.xavier_initializer())
# 将池化输出转换为一维
h_pool1_flat = tf.reshape(pool4, [-1, 47*49*256])
# 激活函数
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

# Dropout层，避免过拟合
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

# 输出层-动作类型
w_actiontype = tf.get_variable('w_actiontype', shape=[1024, 2], initializer=tf.contrib.layers.xavier_initializer())
b_actiontype = tf.get_variable('b_actiontype', shape=[2], initializer=tf.contrib.layers.xavier_initializer())
layer_p = tf.matmul(h_fc1_drop, w_actiontype) + b_actiontype
layer_softmax = tf.nn.softmax(layer_p)
layer_output = tf.argmax(layer_softmax, 1)

y_input = tf.placeholder(shape=[None], dtype=tf.int32)
y_ = tf.reshape(y_input, shape=[-1, 1])
y_onehot = tf.one_hot(y_, depth=2)

# 成本函数 reduce mean 降维->平均值
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer_p, labels=y_onehot))
# 使用了Adam算法来最小化成本函数
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    cost_all = 0.0
    for e in range(50000):
        xs, ys = get_data()
        cost_, _ = sess.run([cost, optimizer], feed_dict={envInput: xs, y_input: ys, keep_prob: 0.3})
        cost_all += cost_
        if e % 100 == 0:
            print('Steps: ', e, 'Cost:', cost_all / 100)
            xs, ys = get_data(batch_size=50, is_train=False)
            cost_ = sess.run(cost, feed_dict={envInput: xs, y_input: ys, keep_prob: 1})
            print('Test Cost:', cost_)
            cost_all = 0.0

        # Checking
        if (e + 1) % 1000 == 0:
            check_x, ReferenceSet = get_check_data()

            PredictionSet = np.zeros([1000], dtype=np.int32)
            for i in range(20):
                start_i = i * 50
                end_i = start_i + 50
                xs = check_x[start_i:end_i]

                out = sess.run(layer_output, feed_dict={envInput: xs, keep_prob: 1})
                PredictionSet[start_i:end_i] = out

            cross = float(sum(PredictionSet*ReferenceSet))
            precision = cross / (float(sum(PredictionSet)) + 0.000001)
            recall = cross / (float(sum(ReferenceSet)) + 0.000001)
            f1 = (2.0 * precision * recall) / (precision + recall + 0.000001)
            print('-= F1 =-: ', f1)


    print('Start forecast...')

    test_data = np.load('/media/xy/247E930D7E92D740/ShareData/test_data.npy')

    PredictionSet = np.zeros([4000], dtype=np.int32)
    for i in range(80):
        start_i = i * 50
        end_i = start_i + 50
        xs = test_data[start_i:end_i]

        out = sess.run(layer_output, feed_dict={envInput: xs, keep_prob: 1})
        PredictionSet[start_i:end_i] = out

    #for label in PredictionSet:
    #    print(label)
    print(PredictionSet)