# coding:utf-8
import numpy as np
from AirLine_Phase_I.DataReader import DataReader
from AirLine_Phase_I.Environment import Environment
import tensorflow as tf
import datetime

# 训练总代数
total_episodes = 2
# 损失参数
loss_param = np.array([100000, 0.5, 0.1, 0.1, 0.075, 0.01, 0.015] ,dtype=np.float32)
# 最大允许的空飞航班数
max_emptyflights = 100
# 每一轮调整允许的最大步数
max_steps = 2500

# 数据读取器
reader = DataReader(filename='DATA_20170705.xlsx' , env_d=68)
#reader.read(is_save=True, filename='env.npy')
env, fault, df_special_passtime = reader.read_fromfile(filename='env.npy')

# 环境控制对象
envObj = Environment(reader.arr_env, max_steps, max_emptyflights, fault,
                     reader.df_fault, reader.df_limit, reader.df_close, reader.df_flytime, reader.base_date,
                     reader.df_plane_type, reader.df_first, reader.df_last, df_special_passtime,
                     reader.domestic_airport)

# 环境长度(航班总数+可能的最大空飞航班数)
env_len = env.shape[0] + max_emptyflights
# 环境维度
env_d = env.shape[1]
# 隐含层节点数
H = 500
batch_size = 25
learning_rate = 0.01

# 各个输出层维度
# 动作类型
d_action_types = 4  #0调机，1取消，2换飞机，3调整时间(第一阶段没有符合联程拉直的航班)
# 机场
d_airports = len(reader.all_airports)
# 航班ID (仅针对原有航班ID范围，新增的调机航班无需再次处理)
d_lineid = len(env)
# 飞机
d_plane = len(reader.all_plane)
# 提前或者延误多少时间 -6小时~+36小时
d_time_diff = (36 + 6) * 60
# 起飞时间范围
d_time_d = reader.time_d_max - reader.time_d_min

######定义模型######
# 输入层(环境)
input_x = tf.placeholder(tf.float32, [None, env_len * env_d], name='input_x')

# 隐含层
# 权重
w_hidden = tf.get_variable('w_hidden', shape=[env_len * env_d, H], initializer=tf.contrib.layers.xavier_initializer())
# 偏置
b_hidden = tf.get_variable('b_hidden', shape=[H], initializer=tf.contrib.layers.xavier_initializer())
# 隐含层
layer_hidden = tf.nn.relu(tf.matmul(input_x, w_hidden) + b_hidden)

# 输出层-动作类型
w_actiontype = tf.get_variable('w_actiontype', shape=[H, d_action_types], initializer=tf.contrib.layers.xavier_initializer())
b_actiontype = tf.get_variable('b_actiontype', shape=[d_action_types], initializer=tf.contrib.layers.xavier_initializer())
layer_actiontype_p = tf.matmul(layer_hidden, w_actiontype) + b_actiontype
layer_actiontype = tf.nn.softmax(layer_actiontype_p)
actiontype_output = tf.argmax(layer_actiontype, 1)

# 输出层-机场-起飞
w_airport_d = tf.get_variable('w_airport_d', shape=[H, d_airports], initializer=tf.contrib.layers.xavier_initializer())
b_airport_d = tf.get_variable('b_airport_d', shape=[d_airports], initializer=tf.contrib.layers.xavier_initializer())
layer_airport_d = tf.nn.softmax(tf.matmul(layer_hidden, w_airport_d) + b_airport_d)
airport_d_output = tf.argmax(layer_airport_d, 1)

# 输出层-机场-降落
w_airport_a = tf.get_variable('w_airport_a', shape=[H, d_airports], initializer=tf.contrib.layers.xavier_initializer())
b_airport_a = tf.get_variable('b_airport_a', shape=[d_airports], initializer=tf.contrib.layers.xavier_initializer())
layer_airport_a = tf.nn.softmax(tf.matmul(layer_hidden, w_airport_a) + b_airport_a)
airport_a_output = tf.argmax(layer_airport_a, 1)

# 输出层-航班ID
w_line = tf.get_variable('w_line', shape=[H, d_lineid], initializer=tf.contrib.layers.xavier_initializer())
b_line = tf.get_variable('b_line', shape=[d_lineid], initializer=tf.contrib.layers.xavier_initializer())
layer_line = tf.nn.softmax(tf.matmul(layer_hidden, w_line) + b_line)
line_output = tf.argmax(layer_line, 1)

# 输出层-飞机
w_plane = tf.get_variable('w_plane', shape=[H, d_plane], initializer=tf.contrib.layers.xavier_initializer())
b_plane = tf.get_variable('b_plane', shape=[d_plane], initializer=tf.contrib.layers.xavier_initializer())
layer_plane = tf.nn.softmax(tf.matmul(layer_hidden, w_plane) + b_plane)
plane_output = tf.argmax(layer_plane, 1)

# 提前或者延误多少时间
w_time_diff = tf.get_variable('w_time_diff', shape=[H, d_time_diff], initializer=tf.contrib.layers.xavier_initializer())
b_time_diff = tf.get_variable('b_time_diff', shape=[d_time_diff], initializer=tf.contrib.layers.xavier_initializer())
layer_time_diff = tf.nn.softmax(tf.matmul(layer_hidden, w_time_diff) + b_time_diff)
time_diff_output = tf.argmax(layer_time_diff, 1)

# 起飞时间-仅供调机用
w_time_d = tf.get_variable('w_time_d', shape=[H, d_time_d], initializer=tf.contrib.layers.xavier_initializer())
b_time_d = tf.get_variable('b_time_d', shape=[d_time_d], initializer=tf.contrib.layers.xavier_initializer())
layer_time_d= tf.nn.softmax(tf.matmul(layer_hidden, w_time_d) + b_time_d)
time_d_output = tf.argmax(layer_time_d, 1)

# Loss计算值输入
input_y = tf.placeholder(tf.float32, [None, 7], name='input_y')
# 计算Loss用的参数
input_param = tf.placeholder(tf.float32, [7], name='input_param')

loss_ = tf.reshape(tf.reduce_sum(input_y * input_param, 1), [-1, 1])
loglik = tf.log(tf.reduce_sum(layer_actiontype_p * layer_actiontype))
loss = tf.reduce_mean(loss_ * loglik)

# 所有可训练的参数
tvars = tf.trainable_variables()
newGrads = tf.gradients(loss, tvars)

# 优化器使用Adam算法
adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
# 所有网络参数的梯度placeholder
W1Grad = tf.placeholder(tf.float32, name='batch_w_grad1')
W2Grad = tf.placeholder(tf.float32, name='batch_w_grad2')
W3Grad = tf.placeholder(tf.float32, name='batch_w_grad3')
W4Grad = tf.placeholder(tf.float32, name='batch_w_grad4')
W5Grad = tf.placeholder(tf.float32, name='batch_w_grad5')
W6Grad = tf.placeholder(tf.float32, name='batch_w_grad6')
W7Grad = tf.placeholder(tf.float32, name='batch_w_grad7')
W8Grad = tf.placeholder(tf.float32, name='batch_w_grad8')

B1Grad = tf.placeholder(tf.float32, name='batch_b_grad1')
B2Grad = tf.placeholder(tf.float32, name='batch_b_grad2')
B3Grad = tf.placeholder(tf.float32, name='batch_b_grad3')
B4Grad = tf.placeholder(tf.float32, name='batch_b_grad4')
B5Grad = tf.placeholder(tf.float32, name='batch_b_grad5')
B6Grad = tf.placeholder(tf.float32, name='batch_b_grad6')
B7Grad = tf.placeholder(tf.float32, name='batch_b_grad7')
B8Grad = tf.placeholder(tf.float32, name='batch_b_grad8')

batchGrad = [W1Grad, B1Grad, W2Grad, B2Grad, W3Grad, B3Grad, W4Grad, B4Grad
    , W5Grad, B5Grad, W6Grad, B6Grad, W7Grad, B7Grad, W8Grad, B8Grad, ]
updateGrads = adam.apply_gradients(zip(batchGrad, tvars))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    xs = []
    ys = []

    x = envObj.env.copy().astype(np.float32)
    x = np.reshape(x, [-1])
    xs.append(x)
    #ys.append(np.array([610., 0., 5500., 0., 0., 0., 0.], dtype=np.float32))
    ys.append(np.array([611., 60., 0., 0., 0., 0., 0.], dtype=np.float32))
    print(ys)
    o1_, o2_, o3_ = sess.run([loss_, loglik, loss], feed_dict={input_x: xs, input_y: ys, input_param: loss_param})
    print(o1_)
    print(o2_)
    print(o3_)

    '''
    gradBuffer = sess.run(tvars)
    print('gradBuffer', np.array(gradBuffer).shape)
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0
    '''
    episode_number = 10
    while episode_number <= total_episodes:
        episode_number += 1

        print(episode_number, 'Start')
        end = False
        # 重置环境
        envObj.reset()
        # 保存初始环境与Loss计算值
        x = envObj.env.copy().astype(np.float32)
        x = np.reshape(x, [-1])
        xs.append(x)
        ys.append(envObj.loss_val)
        while end is False:
            # 预测并执行动作(只用当前一个环境作为输入)
            # 预测
            actiontype_, airport_d_, airport_a_, line_, plane_, time_diff_, time_d_ = \
                sess.run([actiontype_output,
                      airport_d_output,
                      airport_a_output,
                      line_output,
                      plane_output,
                      time_diff_output,
                      time_d_output],
                     feed_dict={input_x: [x]})
            # 0调机，1取消，2换飞机，3调整时间(第一阶段没有符合联程拉直的航班)
            # 调机
            action = np.array([0, 1, 0, 0, 0, 0])
            if actiontype_[0] == 0:
                # 构建动作数组
                action[0] = 0
                action[1] = 0
                action[2] = reader.all_airports[airport_d_[0]]
                action[3] = reader.all_airports[airport_a_[0]]
                action[4] = time_d_[0]
                action[5] = reader.all_plane[plane_[0]]
            # 取消
            elif actiontype_[0] == 1:
                # 构建动作数组
                action[0] = line_[0]
                action[1] = 1
                action[2] = 0
                action[3] = 0
                action[4] = 0
                action[5] = 0
            # 换飞机
            elif actiontype_[0] == 2:
                # 构建动作数组
                action[0] = line_[0]
                action[1] = 2
                action[2] = 0
                action[3] = 0
                action[4] = time_diff_[0] - (6 * 60)
                action[5] = reader.all_plane[plane_[0]]
            # 调整时间
            elif actiontype_[0] == 3:
                # 构建动作数组
                action[0] = line_[0]
                action[1] = 3
                action[2] = 0
                action[3] = 0
                action[4] = time_diff_[0] - (6 * 60)
                action[5] = 0

            #action = np.array([1, 3, 0, 0, 55, 0])
            #print('befor', envObj.env[0])
            end = envObj.step(action)
            print(episode_number, envObj.action_count, action, envObj.loss_val)

            #print('after', envObj.env[0])
            # 保存动作执行后的环境与Loss计算值
            x = envObj.env.copy().astype(np.float32)
            x = np.reshape(x, [-1])
            xs.append(x)
            ys.append(envObj.loss_val)
        print(episode_number, 'End')

        #print(episode_number, 'Run newGrads')
        #tGrad = sess.run(newGrads, feed_dict={input_x: xs, input_y: ys, input_param: loss_param})
        #print('gradBuffer', np.array(tGrad).shape)
        #for ix, grad in enumerate(tGrad):
        #    gradBuffer[ix] += grad

        #if episode_number % batch_size == 0:
        #    print(episode_number, 'Start update grad')
        #    sess.run(updateGrads, feed_dict={W1Grad: gradBuffer[0], W2Grad: gradBuffer[1]})
        #    for ix, grad in enumerate(gradBuffer):
        #        gradBuffer[ix] = grad * 0

