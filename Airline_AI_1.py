# coding:utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow.contrib as tfc
import time

airport_list = ['北京', '上海', '武汉', '巴黎', '伦敦', '拉萨', '长沙', '香港', '厦门', '深圳']
airport_id_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# 环境维度
env_d = 35

f = open('航班表.csv')
df = pd.read_csv(f)
df = df.fillna(value=1.0)
df = df.sort_values(by='航班ID', ascending=True)


# 合并上后继航班
df_merged = pd.merge(df, df,how='left' ,
                     left_on=['日期', '国际/国内', '到达机场', '飞机ID'],
                     right_on=['日期', '国际/国内', '起飞机场', '飞机ID'])

df_env = df.copy()

#print(df_merged['起飞时间_y'] - df_merged['到达时间_x'])

# 生成默认环境
# 0航班ID，1日期(相对于基准日期的分钟)，2国内/国际，3航班号，4起飞机场，5到达机场，
# 6起飞时间(相对于基准时间的分钟数)，7起飞时间(相对于0点的分钟数)，8到达时间(相对于基准时间的分钟数)，9到达时间(相对于0点的分钟数)
# 10飞机ID，11机型，12重要系数(*10 取整数)
# 13起飞故障，14降落故障，
# 15起飞机场关闭(相对于0点的分钟数)，16起飞机场开放(相对于0点的分钟数)，17起飞机场关闭起效日期，18起飞机场关闭失效日期，19是否起飞机场关闭
# 20降落机场关闭(相对于0点的分钟数)，21降落机场开放(相对于0点的分钟数)，22降落机场关闭起效日期，23降落机场关闭失效日期，24是否降落机场关闭
# 25是否飞机限制，26先导航班ID，27后继航班ID，28过站时间(分钟数)
# 29是否取消(0-不取消，1-取消)，30改变航班绑定的飞机(0-不改变，1-改为同型号其他飞机，2-改为不同型号飞机)，
# 31修改航班起飞时间(0-修改，1-不修改)，32联程拉直(0-不拉直，1-拉直，注：第一段设置为拉直后第二段状态为取消，或者用其他方式处理)，
# 33调机(0-不调，1-调)，34时间调整量(分钟数)
arr_env = np.zeros([len(df), env_d], dtype=np.int32)

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
arr_env[:,7] = ds.dt.hour * 60 + ds.dt.minute

# 到达时间
ds = pd.to_datetime(df['到达时间'])
ds_days = (ds - base_date).dt.days
ds_seconds = (ds - base_date).dt.seconds
ds_minutes_e = (ds_days * 24 * 60) + (ds_seconds / 60)
arr_env[:,8] = ds_minutes_e
arr_env[:,9] = ds.dt.hour * 60 + ds.dt.minute

# 飞机ID
arr_env[:,10] = df['飞机ID']

# 机型
arr_env[:,11] = df['机型']

# 重要系数
arr_env[:,12] = df['重要系数'] * 10

# 先导航班ID
id_fw = pd.merge(df[['航班ID']], df_merged[['航班ID_x', '航班ID_y']], how='left', left_on='航班ID', right_on='航班ID_y')
arr_env[:,26] = id_fw['航班ID_x'].fillna(value=0.0)

# 后继航班ID
id_next = df_merged['航班ID_y'].fillna(value=0.0)
arr_env[:,27] = id_next

# 过站时间
# 同一架飞机，无论是否联程航班
ds_d = pd.to_datetime(df_merged['起飞时间_y'])
ds_a = pd.to_datetime(df_merged['到达时间_x'])
ds_os = ((ds_d - ds_a).dt.days * 24 * 60) + ((ds_d - ds_a).dt.seconds / 60)
ds_os = ds_os.fillna(value=999.0)
arr_env[:,28] = ds_os

f_fault = open('故障表.csv')
df_fault = pd.read_csv(f_fault)
df_fault = df_fault.fillna(value=-1)
df_fault['机场'] = df_fault['机场'].replace(airport_list, airport_id_list)

ds_s = pd.to_datetime(df_fault['开始时间'])
ds_e = pd.to_datetime(df_fault['结束时间'])

ds_s_days = (ds_s - base_date).dt.days
ds_s_seconds = (ds_s - base_date).dt.seconds
ds_s_minutes = (ds_s_days * 24 * 60) + (ds_s_seconds / 60)

ds_e_days = (ds_e - base_date).dt.days
ds_e_seconds = (ds_e - base_date).dt.seconds
ds_e_minutes = (ds_e_days * 24 * 60) + (ds_e_seconds / 60)

df_fault['开始时间'] = ds_s_minutes
df_fault['结束时间'] = ds_e_minutes

# 航班飞机限制表
f_limit = open('航线-飞机限制表.csv')
df_limit = pd.read_csv(f_limit)
df_limit['起飞机场'] = df_limit['起飞机场'].replace(airport_list, airport_id_list)
df_limit['到达机场'] = df_limit['到达机场'].replace(airport_list, airport_id_list)

# 机场关闭限制表
f_close = open('机场关闭限制表.csv')
df_close = pd.read_csv(f_close)
df_close['机场'] = df_close['机场'].replace(airport_list, airport_id_list)

ds = pd.to_datetime(df_close['生效日期'])
ds = (ds - base_date).dt.days
df_close['生效日期'] = ds * 24 * 60
ds = pd.to_datetime(df_close['失效日期'])
ds = (ds - base_date).dt.days
df_close['失效日期'] = ds * 24 * 60

ds = pd.to_datetime(df_close['关闭时间'])
df_close['关闭时间'] = ds.dt.hour * 60 + ds.dt.minute
ds = pd.to_datetime(df_close['开放时间'])
df_close['开放时间'] = ds.dt.hour * 60 + ds.dt.minute

print(df_close)


# 附加状态的处理
# 故障、航线-飞机限制、机场关闭限制
def app_action(env = np.zeros([0, env_d], dtype=np.int32), action = np.zeros([2], dtype=np.int32)):
    for row in env:
        ###############################################################################################################
        # 故障航班
        line_id = row[0] # 航班ID
        airport_d = row[4] # 起飞机场
        airport_a = row[5]  # 到达机场
        time_d = row[6]  # 起飞时间
        time_a = row[8]  # 到达时间
        time_d_0 = row[7]  # 起飞时间，0点为基准
        time_a_0 = row[8]  # 到达时间，0点为基准
        plane_id = row[10] # 飞机ID
        # 起飞故障(起飞时间在范围内 & 故障类型=飞行 & (起飞机场相同 | 故障机场为空) & (航班ID相同 | 航班ID为空) & (飞机ID相同 | 飞机ID为空))
        r_ = len(df_fault[((time_d >= df_fault['开始时间']) & (time_d <= df_fault['结束时间']))
                          & (df_fault['故障类型'] == '飞行')
                          & ((df_fault['机场'] == airport_d) | (df_fault['机场'] == -1))
                          & ((df_fault['航班ID'] == line_id) | (df_fault['航班ID'] == -1))
                          & ((df_fault['飞机'] == plane_id) | (df_fault['飞机'] == -1))
                 ])
        if r_ > 0 :
            row[13] = 1

        # 降落故障(到达时间在范围内 & 故障类型=飞行|降落 & (到达机场相同 | 故障机场为空) & (航班ID相同 | 航班ID为空) & (飞机ID相同 | 飞机ID为空))
        r_ = len(df_fault[((time_a >= df_fault['开始时间']) & (time_a <= df_fault['结束时间']))
                          & ((df_fault['故障类型'] == '飞行') | (df_fault['故障类型'] == '降落'))
                          & ((df_fault['机场'] == airport_a) | (df_fault['机场'] == -1))
                          & ((df_fault['航班ID'] == line_id) | (df_fault['航班ID'] == -1))
                          & ((df_fault['飞机'] == plane_id) | (df_fault['飞机'] == -1))
                 ])
        if r_ > 0 :
            row[14] = 1

        ###############################################################################################################
        # 是否航线-飞机限制
        # 起降机场一致、飞机ID一致
        r_ = len(df_limit[(df_limit['起飞机场'] == airport_d)
                          & (df_limit['到达机场'] == airport_a)
                          & (df_limit['限制飞机'] == plane_id)])
        if r_ > 0 :
            row[25] = 1

        ###############################################################################################################
        # 机场关闭
        # 起飞机场关闭
        # 起飞机场ID一致 & 起飞时间在机场关闭的生效与失效日期之内
        rows = np.array(df_close[(df_close['机场'] == airport_d)
                        & (time_d >= df_close['生效日期'])
                        & (time_d <= df_close['失效日期'])])
        for row_ in rows:
            # 关闭和开放时间之间跨越24点的处理
            if row_[2] < row_[1]:
                row_[2] += 24 * 60

            row[15] = row_[1]
            row[16] = row_[2]
            row[17] = row_[3]
            row[18] = row_[4]
            print(time_d_0)
            print(row_[1])
            print(row_[2])
            if (time_d_0 >= row_[1]) & (time_d_0 <= row_[2]):

                print('out of time')
                row[19] = 1

        # 降落机场关闭
        # 降落机场ID一致 & 降落时间在机场关闭的生效与失效日期之内
        rows = np.array(df_close[(df_close['机场'] == airport_a)
                        & (time_a >= df_close['生效日期'])
                        & (time_a <= df_close['失效日期'])])
        for row_ in rows:
            # 关闭和开放时间之间跨越24点的处理
            if row_[2] < row_[1]:
                row_[2] += 24 * 60

            row[19] = row_[1]
            row[20] = row_[2]
            row[21] = row_[3]
            row[22] = row_[4]
            if (time_a_0 >= row_[1]) & (time_a_0 <= row_[2]):
                row[24] = 1

    return env

# 测试，加入故障信息

# 统计loss所需各航班数
def loss_airline_count(env = np.zeros([0, 25], dtype=np.int32)):
    # 计算loss时所需的各调整航班数统计
    # 0航站衔接(或者同一架飞机出现在同一时间的其他航线上、联程航班必须使用同一架飞机)=10000，
    # 1航线飞机限制=10000，2机场关闭=10000，3飞机过站时间=10000
    #
    airline_count = np.zeros([10], dtype=np.int32)

    return airline_count


print(app_action(env=arr_env))

# 网络参数
# 隐含层节点数
H = 50
batch_size = 25
learning_rate = 1e-1
# 环境信息维度
D = 25
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

