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

reader = DataReader(filename='DATA_20170705.xlsx' , env_d=68)

#reader.read(is_save=True, filename='env.npy')
env, fault, df_special_passtime = reader.read_fromfile(filename='env.npy')

envObj = Environment(reader.arr_env, max_steps, max_emptyflights, fault,
                     reader.df_fault, reader.df_limit, reader.df_close, reader.df_flytime, reader.base_date,
                     reader.df_plane_type, reader.df_first, reader.df_last, df_special_passtime,
                     reader.domestic_airport)

# 环境长度(航班总数+可能的最大空飞航班数)
env_len = env.shape[0] + max_emptyflights
# 环境维度
env_d = env.shape[1]
