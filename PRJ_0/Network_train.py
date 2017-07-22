import numpy as np
import pandas as pd
import datetime
import tensorflow as tf
import tensorflow.contrib as tfc

print(datetime.datetime.now(), 'Start - Load Data')
train_data = np.load('/media/xy/247E930D7E92D740/ShareData/train_data.npy')
print(datetime.datetime.now(), 'End - Load Data')

df_label = pd.read_csv('df_id_train.csv',header=-1, encoding='utf-8')
arr_label = np.array(df_label)

def print_activations(t):
    print(t.op.name, '' , t.get_shape().as_list())

env_len = 1415
env_d = 61
learning_rate = 0.001

envInput = tf.placeholder(shape=[None, env_len, env_d], dtype=tf.float32)
envIn = tf.reshape(envInput, shape=[-1, env_len, env_d, 1])

conv1 = tfc.layers.convolution2d(inputs=envIn,
                                 num_outputs=32,
                                 kernel_size=[8,8],
                                 stride=[4,4],
                                 padding='VALID',
                                 biases_initializer=None)
conv2 = tfc.layers.convolution2d(inputs=conv1,
                                 num_outputs=64,
                                 kernel_size=[4,4],
                                 stride=[2,2],
                                 padding='VALID',
                                 biases_initializer=None)
conv3 = tfc.layers.convolution2d(inputs=conv2,
                                 num_outputs=64,
                                 kernel_size=[3,3],
                                 stride=[1,1],
                                 padding='VALID',
                                 biases_initializer=None)


pool1 = tfc.layers.max_pool2d(inputs=conv3, kernel_size=[3, 3], stride=[1,1], padding='VALID')

# 全连接层
# 权重
W_fc1 = tf.get_variable('W_fc1', shape=[171*2*64, 1024], initializer=tf.contrib.layers.xavier_initializer())
# 偏置
b_fc1 = tf.get_variable('b_fc1', shape=[1024], initializer=tf.contrib.layers.xavier_initializer())
# 将池化输出转换为一维
h_pool1_flat = tf.reshape(pool1, [-1, 171*2*64])
# 激活函数
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)


# 输出层-动作类型
w_actiontype = tf.get_variable('w_actiontype', shape=[1024, 2], initializer=tf.contrib.layers.xavier_initializer())
b_actiontype = tf.get_variable('b_actiontype', shape=[2], initializer=tf.contrib.layers.xavier_initializer())
layer_actiontype_p = tf.matmul(h_fc1, w_actiontype) + b_actiontype
layer_actiontype = tf.nn.sigmoid(layer_actiontype_p)
actiontype_output = tf.argmax(layer_actiontype, 1)

y_input = tf.placeholder(shape=[None, 1], dtype=tf.int32)
y_onehot = tf.one_hot(y_input, depth=2)

# 成本函数 reduce mean 降维->平均值
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer_actiontype_p, labels=y_onehot))
# 使用了Adam算法来最小化成本函数
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

    for i in range(400):
        i_start = 10 * i
        i_end = 10 * i + 10
        xs = train_data[i_start:i_end]
        ys = np.reshape(arr_label[i_start:i_end, 1], [-1, 1])

        o1_, o2_ = sess.run([cost, optimizer], feed_dict={envInput: xs, y_input: ys})
        print('cost: ', o1_)