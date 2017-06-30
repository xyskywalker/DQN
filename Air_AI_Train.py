# coding:utf-8
import numpy as np
#import tensorflow as tf
#import matplotlib.pyplot as plt
import pandas as pd
#import tensorflow.contrib as tfc

# 环境维度
env_d = 35

# f = open('航班表.csv')
# df = pd.read_csv(f)
# df = df.fillna(value=1.0)
# df = df.sort_values(by='航班ID', ascending=True)

df_excel = pd.read_excel('DATA_20170627.xlsx', sheetname=[0, 1, 2, 3, 4, 5])

df = df_excel[0].fillna(value=1.0)
df = df.sort_values(by='航班ID', ascending=True)


#print(df_merged['起飞时间_y'] - df_merged['到达时间_x'])

# 生成默认环境
#
# 0~12 基本信息
# 0航班ID，1日期(相对于基准日期的分钟)，2国内/国际，3航班号，4起飞机场，5到达机场，
# 6起飞时间(相对于基准时间的分钟数)，7起飞时间(相对于0点的分钟数)，8到达时间(相对于基准时间的分钟数)，9到达时间(相对于0点的分钟数)
# 10飞机ID，11机型，12重要系数(*10 取整数)
#
# 13~30 故障信息
# 13、14、15起飞机场故障(状态、开始时间、结束时间)，16、17、18降落机场故障(状态、开始时间、结束时间)，
# 19、20、21航班故障(状态、开始时间、结束时间)，
# 22、23、24飞机故障(状态、开始时间、结束时间)，
# 25、26、27起飞机场停机限制(状态、开始时间、结束时间)，28、29、30降落机场停机限制(状态、开始时间、结束时间)
#
# 31~40 机场关闭信息
# 31起飞机场关闭(相对于0点的分钟数)，32起飞机场开放(相对于0点的分钟数)，33起飞机场关闭起效日期，34起飞机场关闭失效日期，35是否起飞机场关闭
# 36降落机场关闭(相对于0点的分钟数)，37降落机场开放(相对于0点的分钟数)，38降落机场关闭起效日期，39降落机场关闭失效日期，40是否降落机场关闭
#
# 41~48 先导、后继、过站时间、联程、中转等信息
# 41先导航班ID，42后继航班ID，43过站时间(分钟数)、44是否联程航班，45联程航班ID
# 46是否有中转、47中转类型(国内-国内:0、国内-国际:1、国际-国内:2、国际-国际:3)、48中转时间限制
#
# 49~54 调整方法信息
# 49是否取消(0-不取消，1-取消)，50改变航班绑定的飞机(0-不改变，1-改为同型号其他飞机，2-改为不同型号飞机)，
# 51修改航班起飞时间(0-不修改，1-延误、2-提前)，52联程拉直(0-不拉直，1-拉直，注：第一段设置为拉直后第二段状态为取消，或者用其他方式处理)，
# 53调机(0-不调，1-调)，54时间调整量(分钟数)
#
# 55~ 航线-飞机限制
# 动态添加，长度为 df_limit.groupby(['起飞机场', '降落机场']).count().max()
#
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

# 起飞/降落机场
arr_env[:,4] = df['起飞机场']
arr_env[:,5] = df['降落机场']

# 起飞时间
ds = pd.to_datetime(df['起飞时间'])
ds_days = (ds - base_date).dt.days
ds_seconds = (ds - base_date).dt.seconds
ds_minutes_s = (ds_days * 24 * 60) + (ds_seconds / 60)
arr_env[:,6] = ds_minutes_s
arr_env[:,7] = ds.dt.hour * 60 + ds.dt.minute

# 到达时间
ds = pd.to_datetime(df['降落时间'])
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


# 故障表
df_fault = df_excel[5]
df_fault = df_fault.fillna(value=-1)

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
df_limit = df_excel[1]

# 机场关闭限制表
df_close = df_excel[2]
ds = pd.to_datetime(df_close['生效日期'])
ds = (ds - base_date).dt.days
df_close['生效日期'] = ds * 24 * 60
ds = pd.to_datetime(df_close['失效日期'])
ds = (ds - base_date).dt.days
df_close['失效日期'] = ds * 24 * 60

ds = pd.to_datetime(df_close['关闭时间'], format='%H:%M:%S')
df_close['关闭时间'] = ds.dt.hour * 60 + ds.dt.minute
ds = pd.to_datetime(df_close['开放时间'], format='%H:%M:%S')
df_close['开放时间'] = ds.dt.hour * 60 + ds.dt.minute


# 附加状态的处理
# 故障、航线-飞机限制、机场关闭限制
def app_action(env = np.zeros([0, env_d], dtype=np.int32), action = np.zeros([2], dtype=np.int32)):
    for row in env:
        line_id = row[0] # 航班ID
        airport_d = row[4] # 起飞机场
        airport_a = row[5]  # 到达机场
        time_d = row[6]  # 起飞时间
        time_a = row[8]  # 到达时间
        time_d_0 = row[7]  # 起飞时间，0点为基准
        time_a_0 = row[9]  # 到达时间，0点为基准
        plane_id = row[10] # 飞机ID

        ###############################################################################################################
        # 故障航班
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
        #r_ = len(df_limit[(df_limit['起飞机场'] == airport_d)
        #                  & (df_limit['降落机场'] == airport_a)
        #                  & (df_limit['飞机ID'] == plane_id)])
        #if r_ > 0 :
        #    row[25] = 1

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
            if (time_d_0 >= row_[1]) & (time_d_0 <= row_[2]):
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


