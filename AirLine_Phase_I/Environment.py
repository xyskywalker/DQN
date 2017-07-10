import numpy as np
import pandas as pd


class Environment():
    def __init__(self, env, max_actions, max_emptyflights, init_fault, df_fault, df_limit, df_close, df_flytime, base_date):
        self.default_env = env
        self.env = self.default_env.copy()
        self.default_fault = init_fault
        self.fault = self.default_fault

        # 故障表
        self.df_fault = df_fault
        # 航线-飞机限制表
        self.df_limit = df_limit
        # 机场关闭表
        self.df_close = df_close
        # 飞行时间表
        self.df_flytime = df_flytime
        # 基准日期
        self.base_date = base_date

        # Loss 计数及重要系数
        # 0, 100000:失效/故障/台风
        # 1, 50:调机
        # 2, 10:取消
        # 3, 10:机型发生变化
        # 4, 7.5:联程拉直
        # 5, 1:延误
        # 6, 1.5:提前
        self.loss_val = np.zeros([7])
        self.loss_val[0] = self.fault

        # 最大允许调机的数量
        self.max_emptyflights = max_emptyflights
        self.max_emptyflights_count = 0

        # 最大允许动作数量
        self.max_actions = max_actions
        # 动作日志记录
        # 0:航班ID
        # 1:起飞机场
        # 2:降落机场
        # 3:起飞时间
        # 4:降落时间
        # 5:飞机ID
        # 6:是否取消
        # 7:是否拉直
        # 8:是否调机
        self.action_log = np.zeros([self.max_actions, 9])
        # 动作计数器
        self.action_count = 0


    # 环境重置
    def reset(self):
        self.env = self.default_env.copy()
        self.fault = self.default_fault
        self.loss_val = np.zeros([7])
        self.loss_val[0] = self.fault
        self.max_emptyflights_count = 0
        self.action_log = np.zeros([self.max_actions, 9])
        self.action_count = 0


    # action
    # 0: Line ID
    # 1: Action Type: 0调机，1取消，2换飞机，3调整时间，4联程拉直
    # 2: 起飞机场
    # 3: 降落机场
    # 4: 起飞时间
    # 5: 降落时间
    # 6: 飞机ID
    def step(self, action):
        # 结束标识
        end = False

        # 处理不同的action type
        action_type = action[1]
        return_count = 0
        if action_type == 0:
            return_count = self.do_action_emptyflights(action[0])
        elif action_type == 1:
            return_count = self.do_action_cancel(action[0])


        # 操作计数+1
        self.action_count += return_count
        # 达到最大操作数限制时结束
        if self.action_count >= self.max_actions:
            end = True

        return self.env, self.action_log, self.action_count, end

    # 动作处理：取消航班
    # lineID: 需要取消的航班ID
    # 本航班取消，动作日志表添加记录
    # 更新先导后继航班
    # 更新过站时间(取消航班只会让过站时间增加)
    # 暂时保留的硬约束：
    #   取消之后检查先后航站衔接是否发生问题，增加fault计数(默认取消一个航班前后衔接必然发生问题，但是连续取消多个航班可能衔接又正常了)
    # 直接退出的硬约束：
    #   边界禁止调整
    def do_action_cancel(self, lineID):
        # 处理的航班
        row = self.env[lineID - 1]
        if row[52] == 1:
            # 已经取消过的不处理，直接退出
            return 0
        else:
            # 取消标记
            row[52] = 1
            # 先导航班：
            row_pre = self.env[row[43] - 1]
            # 后继航班：
            row_next = self.env[row[44] - 1]
            # 先导航班的后继航班变成本航班的后继航班
            row_pre[44] = row_next[0]
            # 后继航班的先导变成本航班的先导航班
            row_next[43] = row_pre[0]
            # 过站时间更新
            row_pre[45] = row_next[6] - row_pre[8]

            # 取消航班数+1*重要系数
            self.loss_val[2] += 1 * row[12]
            # action 日志更新
            log = self.action_log[self.action_count]
            log[0] = lineID
            log[6] = 1

            # 检测航站衔接
            if row_pre[5] != row_next[4]:
                # 航站衔接有问题，如果之前已经航站衔接fault那么不加，否则fault+1
                if row_pre[58] == 0:
                    self.fault += 1
                    self.loss_val[0] = self.fault
                    row_pre[58] = 1
            else:
                # 航站衔接正常了，那么检测之前是否不正常，如果由之前的不正常变成正常了，fault需要-1
                if row_pre[58] == 1:
                    self.fault -= 1
                    self.loss_val[0] = self.fault
                    row_pre[58] = 0
            return 1

    # 动作处理：换飞机
    # lineID: 需要换飞机的航班ID
    # time_d: 调整后的起飞时间，飞行时间与原飞行时间相同
    # 先导后继航班重算
    # 更新过站时间
    # 暂时保留的硬约束：
    # 直接退出的硬约束：
    def do_action_flightchange(self, lineID, new_planeID, time_d):
        return 1

    # 动作处理：调机
    def do_action_emptyflights(self, lineID):
        return lineID


    def show(self):
        print('loss_val: ', self.loss_val)
        print('fault: ', self.fault)
        print('action_log: ', self.action_log)
        print('action_count: ', self.action_count)

