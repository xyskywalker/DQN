import numpy as np
from AirLine_Phase_I.DataReader import DataReader
from AirLine_Phase_I.Environment import Environment
import datetime
import time
import pandas as pd
import tensorflow as tf
import tensorflow.contrib as tfc


# loss的参数，需要调整的即失效航班使用一个非常大的参数，原目标函数的参数一律除以100处理，用以加大与失效航班的差异
# 0, 100000:失效/故障/台风
# 1, 50:调机
# 2, 10:取消
# 3, 10:机型发生变化
# 4, 7.5:联程拉直
# 5, 1:延误
# 6, 1.5:提前
loss_para = np.array([100000, 50, 10, 10, 7.5, 1, 1.5] ,dtype=np.float32)

reader = DataReader(filename='DATA_20170705.xlsx' , env_d=68)

#reader.read(is_save=True, filename='env.npy')
env, fault, df_special_passtime = reader.read_fromfile(filename='env.npy')

envObj = Environment(reader.arr_env, 2364, 100, fault,
                     reader.df_fault, reader.df_limit, reader.df_close, reader.df_flytime, reader.base_date,
                     reader.df_plane_type, reader.df_first, reader.df_last, df_special_passtime,
                     reader.domestic_airport)

def print_activations(t):
    print(t.op.name, '' , t.get_shape().as_list())

# 环境长度(航班总数+可能的最大空飞航班数)
env_len = env.shape[0] + 100
# 环境维度
env_d = env.shape[1]

envInput = tf.placeholder(shape=[None, env_len, env_d], dtype=tf.float32)
envIn = tf.reshape(envInput, shape=[-1, env_len, env_d, 1])

conv1 = tfc.layers.convolution2d(inputs=envIn,
                                 num_outputs=32,
                                 kernel_size=[200,20],
                                 stride=[4,4],
                                 padding='VALID',
                                 biases_initializer=None)
conv2 = tfc.layers.convolution2d(inputs=conv1,
                                 num_outputs=64,
                                 kernel_size=[100,10],
                                 stride=[2,2],
                                 padding='VALID',
                                 biases_initializer=None)
conv3 = tfc.layers.convolution2d(inputs=conv2,
                                 num_outputs=64,
                                 kernel_size=[50,5],
                                 stride=[1,1],
                                 padding='VALID',
                                 biases_initializer=None)
conv4 = tfc.layers.convolution2d(inputs=conv3,
                                 num_outputs=512,
                                 kernel_size=[10,1],
                                 stride=[1,1],
                                 padding='VALID',
                                 biases_initializer=None)


pool1 = tfc.layers.max_pool2d(inputs=conv4, kernel_size=[1,1], stride=[10,1], padding='VALID')

# 全连接层
# 权重
W_fc1 = tf.get_variable('W_fc1', shape=[18*15*512, 1024], initializer=tf.contrib.layers.xavier_initializer())
# 偏置
b_fc1 = tf.get_variable('b_fc1', shape=[1024], initializer=tf.contrib.layers.xavier_initializer())
# 将第二个卷积层的池化输出转换为一维
h_pool1_flat = tf.reshape(pool1, [-1, 18*15*512])
# 激活函数
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)


# 输出层-动作类型
w_actiontype = tf.get_variable('w_actiontype', shape=[1024, 4], initializer=tf.contrib.layers.xavier_initializer())
b_actiontype = tf.get_variable('b_actiontype', shape=[4], initializer=tf.contrib.layers.xavier_initializer())
layer_actiontype_p = tf.matmul(h_fc1, w_actiontype) + b_actiontype
layer_actiontype = tf.nn.softmax(layer_actiontype_p)
actiontype_output = tf.argmax(layer_actiontype, 1)

print_activations(envInput)
print_activations(envIn)
print_activations(conv1)
print_activations(conv2)
print_activations(conv3)
print_activations(conv4)
print_activations(pool1)
print_activations(h_fc1)
print_activations(layer_actiontype_p)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    o_ = sess.run(actiontype_output, feed_dict={envInput: [envObj.env]})
    print(o_)

    print(np.sum(envObj.env))
    action = np.array([1, 1, 0, 0, 0, 0, 0])
    envObj.step(action=action)
    print(np.sum(envObj.env))

    action = np.array([11, 3, 0, 0, 30, 0, 55])
    envObj.step(action=action)
    print(np.sum(envObj.env))

    o_ = sess.run(actiontype_output, feed_dict={envInput: [envObj.env]})
    print(o_)

#print(reader.time_d_max - reader.time_d_min)
#time_d = 6000

#print('fault', fault)
#print(reader.df_plane_type['飞机ID'].drop_duplicates().count())
# dt = datetime.datetime.fromtimestamp(reader.base_date)

#dt1 = reader.base_date.to_pydatetime()
#dt2 = dt1 + datetime.timedelta(minutes=time_d)
#dt2 = datetime.datetime.fromtimestamp(time.time())

#print(dt2)
#print(dt2.hour)
#print(dt2.minute)
#rint(env[15])
#print(reader.arr_env[10])

#envObj.show()
'''
test_df = pd.DataFrame(env)
print(test_df[((test_df[13] ==1) | (test_df[16] == 1) | (test_df[25] == 1) | (test_df[29] == 1))
              & (test_df[46] == 1)
              & (test_df[2] == 1) & (test_df[5] == 50)])
'''

'''
end = False
i = 1
while end is False:
    print(datetime.datetime.now(), 'Start - Action')
    action = np.array([i, 4, 0, 0, 30, 0, 55])
    _, _ ,_, end = envObj.step(action=action)
    print(datetime.datetime.now(), 'End - Action')
    i += 1
'''

'''
print(datetime.datetime.now(), 'Start - Action')
for i in range(800):
    action = np.array([1, 1, 0, 0, 0, 0, 0])
    envObj.step(action=action)
    action = np.array([11, 2, 0, 0, 30, 0, 55])
    envObj.step(action=action)
    action = np.array([11, 3, 0, 0, 30, 0, 55])
    envObj.step(action=action)

print(datetime.datetime.now(), 'End - Action')


print(datetime.datetime.now(), 'Start - Action')
loss_param = np.array([100000, 0.5, 0.1, 0.1, 0.075, 0.01, 0.015] ,dtype=np.float32)
action = np.array([ 611.,   60.,    0.,    0.,    0.,    0.,    0.])
print(sum(loss_param * action))

#envObj.step(action=action)

print(datetime.datetime.now(), 'End - Action')
print('loss_val: ', envObj.loss_val)
print('fault: ', envObj.fault)
print('action_log: ', envObj.action_log)
print('action_log: ', envObj.action_log[0])
print('action_log: ', envObj.action_log[1])
print('action_count: ', envObj.action_count)

x = []
testarr1 = np.array([[1,2], [3, 4]])
x_ = np.reshape(testarr1, [-1])
print(x_)
x.append(np.array([1,1,1], dtype=np.float32))
x.append(np.array([2,0,2], dtype=np.float32))
print(x)
x = np.vstack(x)
print(np.mean(x, 0))

#envObj.reset()
#envObj.show()


#print(datetime.datetime.now(), 'Start - Read')
#print(reader.read(is_save = True, filename='env.npy'))
#print(datetime.datetime.now(), 'End - Read')

#arr1 = np.random.randint(low=1, high=10, size=[5, 3])
#df = pd.DataFrame(arr1)
#print(df[df[0] > 0].sort_values(by=2, ascending=False))
'''
