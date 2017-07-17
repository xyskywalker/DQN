# coding:utf-8
import numpy as np
from AirLine_Phase_I.DataReader import DataReader
from AirLine_Phase_I.Environment import Environment
import tensorflow as tf
import datetime

# 损失参数
loss_param = np.array([100000, 50, 10, 10, 7.5, 1, 1.5] ,dtype=np.float32)
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
d_plane = reader.df_plane_type['飞机ID'].drop_duplicates().count()
# 提前或者延误多少时间 -6小时~+36小时
d_time_diff = (36 + 6) * 60
# 起飞时间范围
d_time_d = reader.time_d_max - reader.time_d_min

######定义模型######
# 输入层(环境)
input_x = tf.placeholder(tf.int32, [None, env_len, env_d], name='input_x')

# 隐含层
# 权重
w_hidden = tf.get_variable('w_hidden', shape=[env_len, env_d, H], initializer=tf.contrib.layers.xavier_initializer())
# 偏置
b_hidden = tf.get_variable('b_hidden', shape=[env_len, env_d, H], initializer=tf.contrib.layers.xavier_initializer())
# 隐含层
layer_hidden = tf.nn.relu(tf.matmul(input_x, w_hidden) + b_hidden)

# 输出层-动作类型
output_actiontype = 1
