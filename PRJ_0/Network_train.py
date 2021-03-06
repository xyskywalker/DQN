import numpy as np
import pandas as pd
import datetime
import tensorflow as tf
import tensorflow.contrib as tfc

print(datetime.datetime.now(), 'Start - Load Data')
#train_data = np.load('/media/xy/247E930D7E92D740/ShareData/train_data.npy')
train_data = np.load('train_data.npy')
print(datetime.datetime.now(), 'End - Load Data')

df_label = pd.read_csv('df_id_train.csv',header=-1, encoding='utf-8')
arr_label = np.array(df_label)

def print_activations(t):
    print(t.op.name, '' , t.get_shape().as_list())

env_len = 1415
env_d = 61
learning_rate = 0.0001

envInput = tf.placeholder(shape=[None, env_len, env_d], dtype=tf.float32)
envIn = tf.reshape(envInput, shape=[-1, env_len, env_d, 1])

conv1 = tfc.layers.convolution2d(inputs=envIn,
                                 num_outputs=32,
                                 kernel_size=[12, 3],
                                 stride=[8, 2],
                                 padding='VALID',
                                 biases_initializer=None)
conv2 = tfc.layers.convolution2d(inputs=conv1,
                                 num_outputs=64,
                                 kernel_size=[8, 2],
                                 stride=[8, 2],
                                 padding='VALID',
                                 biases_initializer=None)
conv3 = tfc.layers.convolution2d(inputs=conv2,
                                 num_outputs=256,
                                 kernel_size=[3, 3],
                                 stride=[1, 1],
                                 padding='VALID',
                                 biases_initializer=None)


pool1 = tfc.layers.max_pool2d(inputs=conv3, kernel_size=[3, 3], stride=[1, 1], padding='VALID')

# 全连接层
# 权重
W_fc1 = tf.get_variable('W_fc1', shape=[18*11*256, 1024], initializer=tf.contrib.layers.xavier_initializer())
# 偏置
b_fc1 = tf.get_variable('b_fc1', shape=[1024], initializer=tf.contrib.layers.xavier_initializer())
# 将池化输出转换为一维
h_pool1_flat = tf.reshape(pool1, [-1, 18*11*256])
# 激活函数
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

# Dropout层，避免过拟合
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

# 输出层-动作类型
w_actiontype = tf.get_variable('w_actiontype', shape=[1024, 2], initializer=tf.contrib.layers.xavier_initializer())
b_actiontype = tf.get_variable('b_actiontype', shape=[2], initializer=tf.contrib.layers.xavier_initializer())
layer_actiontype_p = tf.matmul(h_fc1_drop, w_actiontype) + b_actiontype
layer_actiontype = tf.nn.softmax(layer_actiontype_p)
actiontype_output = tf.argmax(layer_actiontype, 1)

y_input = tf.placeholder(shape=[None, 1], dtype=tf.int32)
y_onehot = tf.one_hot(y_input, depth=2)

y_o = tf.placeholder(shape=[None], dtype=tf.int64)

cross_count = tf.cast(tf.reduce_sum(tf.multiply(actiontype_output, y_o)), dtype=tf.float32)
cross_zero = tf.greater(cross_count, 0.0)

def fn1():
    return tf.constant(0.0, dtype=tf.float32)

def fn2():
    return tf.constant(1.0, dtype=tf.float32)

f1_ = tf.cond(cross_zero, fn1, fn2, name='f1_')

prediction_count = tf.cast(tf.reduce_sum(actiontype_output), dtype=tf.float32)
reference_count = tf.cast(tf.reduce_sum(y_o), dtype=tf.float32)

precision = tf.div(cross_count, prediction_count + tf.cond(tf.greater(prediction_count, 0.0), fn1, fn2))
recall = tf.div(cross_count, reference_count + tf.cond(tf.greater(reference_count, 0.0), fn1, fn2))

f1 = tf.div(tf.multiply(2.0, tf.multiply(precision, recall)), tf.add(tf.add(precision, recall), f1_))



# 成本函数 reduce mean 降维->平均值
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer_actiontype_p, labels=y_onehot))
# 使用了Adam算法来最小化成本函数
cost2 = (1 - f1) * cost * 10
optimizer2 = tf.train.AdamOptimizer(learning_rate).minimize(cost2)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

print_activations(envInput)
print_activations(envIn)
print_activations(conv1)
print_activations(conv2)
print_activations(conv3)
print_activations(pool1)
print_activations(h_fc1)
print_activations(layer_actiontype_p)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for e in range(10000):
        cost_all = 0.0
        accuracy_all = 0.00
        for i in range(18):
            i_start = 1000 * i
            i_end = 1000 * i + 1000
            xs = train_data[i_start:i_end]
            y_ = arr_label[i_start:i_end, 1]
            ys = np.reshape(y_, [-1, 1])

            o1_, o2_,cross_count_, f1_ = sess.run([cost2, optimizer2, cross_count, f1], feed_dict={envInput: xs, y_input: ys, keep_prob: 0.1, y_o: y_})
            cost_all += o1_
            print('Cost:', o1_, 'F1', f1_, 'cross_count', cross_count_)

        cost_all = cost_all/18.0
        accuracy_all = accuracy_all/18.0

        i_start = 18000
        i_end = i_start + 2000
        xs = train_data[i_start:i_end]
        y_ = arr_label[i_start:i_end, 1]
        ys = np.reshape(y_, [-1, 1])
        print('e', e, 'cost', cost_all)

        t_, o1_, o3_, cross_count_ = sess.run([precision, recall, f1, cross_count], feed_dict={envInput: xs, y_input: ys, keep_prob: 1.0, y_o: y_})
        print('precision: ', t_, ' recall: ', o1_, ' F1: ', o3_, 'cross_count', cross_count_)
