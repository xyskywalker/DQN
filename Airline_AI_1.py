# coding:utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow.contrib as tfc
import time

airport_list = ['北京', '上海', '武汉', '巴黎', '伦敦']
airport_id_list = [0, 1, 2, 3, 4]

f = open('航班表.csv')
df = pd.read_csv(f)
df = df.fillna(value=1.0)
df = df.sort_values(by='日期', ascending=True)

# 生成默认环境
# 0航班ID，1日期(相对于基准日期的分钟)，2国内/国际，3航班号，4起飞机场，5到达机场，
# 6起飞时间(相对于基准时间的分钟数)，7到达时间(相对于基准时间的分钟数)，8飞机ID，9机型，10重要系数(*10 取整数)
# 11起飞故障，12降落故障，13起飞机场关闭，14降落机场关闭，15是否飞机限制，16过站时间(分钟数)
# 17调整方法，18调整量(分钟数)
arr_env = np.zeros([len(df), 19], dtype=np.int32)

# print(df)

# 航班ID
arr_env[:,0] = df['航班ID']

# 日期
ds = pd.to_datetime(df['日期'])
# 基准日期
base_date = ds[0]
ds = (ds - base_date).dt.days
arr_env[:,1] = ds * 24 * 60

# 国际=0/国内=1
arr_env[:,2] = df['国际/国内'].replace(['国际', '国内'], [0, 1])

# 航班号
arr_env[:,3] = df['航班号']

# 起飞/到达机场
arr_env[:,4] = df['起飞机场'].replace(airport_list, airport_id_list)
arr_env[:,5] = df['到达机场'].replace(airport_list, airport_id_list)

# 起飞时间
ds = pd.to_datetime(df['起飞时间'])
ds_days = (ds - base_date).dt.days
ds_seconds = (ds - base_date).dt.seconds
ds_minutes_s = (ds_days * 24 * 60) + (ds_seconds / 60)
arr_env[:,6] = ds_minutes_s

# 到达时间
ds = pd.to_datetime(df['到达时间'])
ds_days = (ds - base_date).dt.days
ds_seconds = (ds - base_date).dt.seconds
ds_minutes_e = (ds_days * 24 * 60) + (ds_seconds / 60)
arr_env[:,7] = ds_minutes_e

# 飞机ID
arr_env[:,8] = df['飞机ID']

# 机型
arr_env[:,9] = df['机型']

# 重要系数
arr_env[:,10] = df['重要系数'] * 10

# 过站时间
arr_env[:,16] = ds_minutes_e - ds_minutes_s

# 测试，加入故障信息

# 统计loss所需各航班数
def loss_airline_count(env = np.zeros([0, 19], dtype=np.int32)):
    # 计算loss时所需的各调整航班数统计
    # 0航站衔接(或者同一架飞机出现在同一时间的其他航线上、联程航班必须使用同一架飞机)=10000，
    # 1航线飞机限制=10000，2机场关闭=10000，3飞机过站时间=10000
    #
    airline_count = np.zeros([10], dtype=np.int32)

    return airline_count

# print(arr_env)

# 网络参数
# 隐含层节点数
H = 50
batch_size = 25
learning_rate = 1e-1
# 环境信息维度
D = 4
# Reward的discount比例
gamma = 0.99


# 定义网络
# 输入
observations = tf.placeholder(tf.float32, [None, D], name='input_x')
W1 = tf.get_variable('W1', shape=[D, H], initializer=tfc.layers.xavier_initializer())
# 隐含层，没有偏置
layer1 = tf.nn.relu(tf.matmul(observations, W1))

W2 = tf.get_variable('W2', shape=[H, 1], initializer=tfc.layers.xavier_initializer())
score = tf.matmul(layer1, W2)
# 输出，往左还是往右
probability = tf.nn.sigmoid(score)

# 优化器使用Adam算法
adam = tf.train.AdamOptimizer(learning_rate=learning_rate)

