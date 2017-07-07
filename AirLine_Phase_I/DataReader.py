import numpy as np
import pandas as pd
import datetime


class DataReader():

    def __init__(self, filename='', env_d=0):
        self.filename = filename
        self.env_d = env_d

        print(datetime.datetime.now(), 'Start - Load Data')
        self.df_excel = pd.read_excel(self.filename, sheetname=[0, 1, 2, 3, 4])

        self.df = self.df_excel[0].fillna(value=1.0)
        self.df = self.df.sort_values(by='航班ID', ascending=True)

        ########################################################################################################################
        # 基准日期
        base_date = self.df['日期'].min()
        # 故障表/台风场景
        self.df_fault = self.df_excel[3]
        self.df_fault = self.df_fault.fillna(value=-1)

        ds_s = pd.to_datetime(self.df_fault['开始时间'])
        ds_e = pd.to_datetime(self.df_fault['结束时间'])

        ds_s_days = (ds_s - base_date).dt.days
        ds_s_seconds = (ds_s - base_date).dt.seconds
        ds_s_minutes = (ds_s_days * 24 * 60) + (ds_s_seconds / 60)

        ds_e_days = (ds_e - base_date).dt.days
        ds_e_seconds = (ds_e - base_date).dt.seconds
        ds_e_minutes = (ds_e_days * 24 * 60) + (ds_e_seconds / 60)

        self.df_fault['开始时间'] = ds_s_minutes
        self.df_fault['结束时间'] = ds_e_minutes
        # 增加一个 已停机数 字段
        self.df_fault['已停机数'] = 0.0

        # 航班飞机限制表
        self.df_limit = self.df_excel[1]

        # 机场关闭限制表
        self.df_close = self.df_excel[2]
        ds = pd.to_datetime(self.df_close['生效日期'])
        ds = (ds - base_date).dt.days
        self.df_close['生效日期'] = ds * 24 * 60
        ds = pd.to_datetime(self.df_close['失效日期'])
        ds = (ds - base_date).dt.days
        self.df_close['失效日期'] = ds * 24 * 60

        ds = pd.to_datetime(self.df_close['关闭时间'], format='%H:%M:%S')
        self.df_close['关闭时间'] = ds.dt.hour * 60 + ds.dt.minute
        ds = pd.to_datetime(self.df_close['开放时间'], format='%H:%M:%S')
        self.df_close['开放时间'] = ds.dt.hour * 60 + ds.dt.minute

        # 飞行时间表
        self.df_flytime = self.df_excel[4]

        print(datetime.datetime.now(), 'End - Load Data')
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
        # 25、26、27、28起飞机场停机限制(状态、停机数、开始时间、结束时间)，29、30、31、32降落机场停机限制(状态、停机数、开始时间、结束时间)
        #
        # 33~42 机场关闭信息
        # 33起飞机场关闭(相对于0点的分钟数)，34起飞机场开放(相对于0点的分钟数)，35起飞机场关闭起效日期，36起飞机场关闭失效日期，37是否起飞机场关闭
        # 38降落机场关闭(相对于0点的分钟数)，39降落机场开放(相对于0点的分钟数)，40降落机场关闭起效日期，41降落机场关闭失效日期，42是否降落机场关闭
        #
        # 43~50 先导、后继、过站时间、联程、中转等信息
        # 43先导航班ID，44后继航班ID，45过站时间(分钟数)、46是否联程航班，47联程航班ID
        # 48是否有中转、49中转类型(国内-国内:0、国内-国际:1、国际-国内:2、国际-国际:3)、50中转时间限制、51对应的出港航班
        #
        # 52~57 调整方法信息
        # 52是否取消(0-不取消，1-取消)，53改变航班绑定的飞机(0-不改变，1-改为同型号其他飞机，2-改为不同型号飞机)，
        # 54修改航班起飞时间(0-不修改，1-延误、2-提前)，55联程拉直(0-不拉直，1-拉直，注：第一段设置为拉直后第二段状态为取消，或者用其他方式处理)，
        # 56调机(0-不调，1-调)，57时间调整量(分钟数)
        #
        # 58~ 航线-飞机限制
        # 58是否航线-飞机限制
        # 动态添加，长度为 self.df_limit.groupby(['起飞机场', '降落机场']).count().max()
        #

        limit_len = self.df_limit.groupby(['起飞机场', '降落机场']).count().max()[0]
        self.arr_env = np.zeros([len(self.df), self.env_d + limit_len], dtype=np.int32)

        print(datetime.datetime.now(), 'Start - BASIC Data')
        ########################################################################################################################
        # 0~12 基本信息
        # 0航班ID，1日期(相对于基准日期的分钟)，2国内/国际，3航班号，4起飞机场，5到达机场，
        # 6起飞时间(相对于基准时间的分钟数)，7起飞时间(相对于0点的分钟数)，8到达时间(相对于基准时间的分钟数)，9到达时间(相对于0点的分钟数)
        # 10飞机ID，11机型，12重要系数(*10 取整数)
        # 航班ID
        self.arr_env[:, 0] = self.df['航班ID']
        # 日期
        ds = pd.to_datetime(self.df['日期'])
        ds = (ds - base_date).dt.days
        days_minutes = ds * 24 * 60
        self.arr_env[:, 1] = days_minutes
        self.df['日期'] = days_minutes
        # 国际=0/国内=1
        self.arr_env[:, 2] = self.df['国际/国内'].replace(['国际', '国内'], [0, 1])
        # 航班号
        self.arr_env[:, 3] = self.df['航班号']
        # 起飞/降落机场
        self.arr_env[:, 4] = self.df['起飞机场']
        self.arr_env[:, 5] = self.df['降落机场']
        # 起飞时间
        ds = pd.to_datetime(self.df['起飞时间'])
        ds_days = (ds - base_date).dt.days
        ds_seconds = (ds - base_date).dt.seconds
        ds_minutes_s = (ds_days * 24 * 60) + (ds_seconds / 60)
        self.arr_env[:, 6] = ds_minutes_s
        self.df['起飞时间'] = ds_minutes_s
        self.arr_env[:, 7] = ds.dt.hour * 60 + ds.dt.minute
        # 到达时间
        ds = pd.to_datetime(self.df['降落时间'])
        ds_days = (ds - base_date).dt.days
        ds_seconds = (ds - base_date).dt.seconds
        ds_minutes_e = (ds_days * 24 * 60) + (ds_seconds / 60)
        self.arr_env[:, 8] = ds_minutes_e
        self.df['降落时间'] = ds_minutes_e
        self.arr_env[:, 9] = ds.dt.hour * 60 + ds.dt.minute
        # 飞机ID
        self.arr_env[:, 10] = self.df['飞机ID']
        # 机型
        self.arr_env[:, 11] = self.df['机型']
        # 重要系数
        self.arr_env[:, 12] = self.df['重要系数'] * 10

        print(datetime.datetime.now(), 'End - BASIC Data')

    # 附加状态的处理
    # 故障、航线-飞机限制、机场关闭限制
    def app_action(self, env = np.zeros([0, 0], dtype=np.int32)):
        print(datetime.datetime.now(), 'Start - Att Data')
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
            # 13~30 故障信息
            #
            # 13、14、15起飞机场故障(状态、开始时间、结束时间):起飞时间在范围内 & 故障类型=飞行 & 起飞机场相同
            r_ = self.df_fault[((time_d >= self.df_fault['开始时间']) & (time_d <= self.df_fault['结束时间']))
                              & (self.df_fault['影响类型'] == '起飞')
                              & (airport_d == self.df_fault['机场'])]
            if len(r_) > 0 :
                t_s = r_['开始时间'].min()
                t_e = r_['结束时间'].max()
                row[13] = 1
                row[14] = t_s
                row[15] = t_e

            # 16、17、18降落机场故障(状态、开始时间、结束时间):降落时间在范围内 & (故障类型=飞行|降落) & 降落机场相同
            r_ = self.df_fault[((time_a >= self.df_fault['开始时间']) & (time_a <= self.df_fault['结束时间']))
                              & (self.df_fault['影响类型'] == '降落')
                              & (airport_a == self.df_fault['机场'])]
            if len(r_) > 0 :
                t_s = r_['开始时间'].min()
                t_e = r_['结束时间'].max()
                row[16] = 1
                row[17] = t_s
                row[18] = t_e

            # 19、20、21航班故障(状态、开始时间、结束时间):起飞时间在范围内 & 故障类型=飞行 & 航班ID相同
            # 22、23、24飞机故障(状态、开始时间、结束时间):起飞时间在范围内 & 故障类型=飞行 & 飞机ID相同

            # 25、26、27、28起飞机场停机限制(状态、停机限制数量、开始时间、结束时间):起飞时间在范围内 & 故障类型=停机 & 起飞机场相同
            r_ = self.df_fault[((time_d >= self.df_fault['开始时间']) & (time_d <= self.df_fault['结束时间']))
                          & (self.df_fault['影响类型'] == '停机') & (airport_d == self.df_fault['机场'])]

            if len(r_) > 0:
                self.df_fault.loc[r_.index, ['已停机数']] += 1

                p_num = 0
                t_s = r_['开始时间'].min()
                t_e = r_['结束时间'].max()
                row[25] = 1
                row[26] = p_num
                row[27] = t_s
                row[28] = t_e

            # 29、30、31、32降落机场停机限制(状态、停机限制数量、开始时间、结束时间):降落时间在范围内 & 故障类型=停机 & 降落机场相同
            r_ = self.df_fault[((time_a >= self.df_fault['开始时间']) & (time_a <= self.df_fault['结束时间']))
                              & (self.df_fault['影响类型'] == '停机') & (airport_a == self.df_fault['机场'])]

            if len(r_) > 0:
                self.df_fault.loc[r_.index, ['已停机数']] += 1

                p_num = 0
                t_s = r_['开始时间'].min()
                t_e = r_['结束时间'].max()
                row[29] = 1
                row[30] = p_num
                row[31] = t_s
                row[32] = t_e

            ###############################################################################################################
            # 33~42 机场关闭信息
            # 33起飞机场关闭(相对于0点的分钟数)，34起飞机场开放(相对于0点的分钟数)，35起飞机场关闭起效日期，36起飞机场关闭失效日期，37是否起飞机场关闭
            # 起飞机场ID一致 & 起飞时间在机场关闭的生效与失效日期之内
            rows = np.array(self.df_close[(self.df_close['机场'] == airport_d)
                            & (time_d >= self.df_close['生效日期'])
                            & (time_d <= self.df_close['失效日期'])])
            for row_ in rows:
                # 关闭和开放时间之间跨越24点的处理
                if row_[2] < row_[1]:
                    row_[2] += 24 * 60

                row[33] = row_[1]
                row[34] = row_[2]
                row[35] = row_[3]
                row[36] = row_[4]
                if (time_d_0 >= row_[1]) & (time_d_0 <= row_[2]):
                    row[37] = 1

            # 38降落机场关闭(相对于0点的分钟数)，39降落机场开放(相对于0点的分钟数)，40降落机场关闭起效日期，41降落机场关闭失效日期，42是否降落机场关闭
            # 降落机场ID一致 & 降落时间在机场关闭的生效与失效日期之内
            rows = np.array(self.df_close[(self.df_close['机场'] == airport_a)
                            & (time_a >= self.df_close['生效日期'])
                            & (time_a <= self.df_close['失效日期'])])
            for row_ in rows:
                # 关闭和开放时间之间跨越24点的处理
                if row_[2] < row_[1]:
                    row_[2] += 24 * 60

                row[38] = row_[1]
                row[39] = row_[2]
                row[40] = row_[3]
                row[41] = row_[4]
                if (time_a_0 >= row_[1]) & (time_a_0 <= row_[2]):
                    row[42] = 1

            ###############################################################################################################
            # 43~50 先导、后继、过站时间、联程、中转等信息
            #
            # 43先导航班ID，44后继航班ID，45过站时间(分钟数)
            # 后继航班：飞机ID相同 起飞时间大于本航班降落时间 按起飞时间排序之后第一个航班
            r_ = self.df[(self.df['飞机ID'] == plane_id) & (self.df['起飞时间'] > time_d)].sort_values(by='起飞时间', ascending=True)
            if len(r_) > 0:
                # 本航班的后继航班
                next_id = r_.iloc[0][0]
                row[44] = next_id
                # 本航班的到达时间
                time_a_ = row[8]
                # 后继航班的先导航班即是本航班
                row_next = env[next_id - 1]
                row_next[43] = row[0]
                # 后继航班的起飞时间
                time_d_ = row_next[6]
                # 过站时间
                row[45] = time_d_ - time_a_

                # 联程航班: 日期与航班号相同
                # 46是否联程航班，47联程航班ID
                if (row[1] == row_next[1]) & (row[3] == row_next[3]) :
                    row[46] = 1
                    # 联程航班第一段的ID就是后继航班ID
                    row[47] = next_id
                    # 联程航班第二段
                    row_next[46] = 1
                    row_next[47] = row[0]

            ###############################################################################################################
            # 中转航班信息: 添加在有中转的进港航班上
            # 48是否有中转(0非中转、1中转)
            # 49中转类型(国内-国内:0、国内-国际:1、国际-国内:2、国际-国际:3)
            # 50中转时间限制
            # 51对应的出港航班

            ###############################################################################################################
            # 58~ 航线-飞机限制
            # 58是否航线-飞机限制(0不限制，1限制)
            # 动态添加，长度为 self.df_limit.groupby(['起飞机场', '降落机场']).count().max()
            arr_limit = np.array(self.df_limit[(self.df_limit['起飞机场'] == airport_d)
                                          & (self.df_limit['降落机场'] == airport_a)]['飞机ID'])
            if plane_id in arr_limit:
                row[58] = 1
            row[59: 59 + len(arr_limit)] = arr_limit

        print(datetime.datetime.now(), 'End - Att Data')
        return env

    def read(self):
        env_ = self.app_action(self.arr_env)

        return env_

reader = DataReader(filename='DATA_20170705.xlsx' , env_d=59)
print(reader.read())

# r_ = self.df[self.df['飞机ID'] == 30].sort_values(by='起飞时间', ascending=True).reset_index()
# r_c = r_[r_['航班ID'] == 223].index[0]
# print(r_c)
# print(datetime.datetime.now())