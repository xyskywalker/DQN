import numpy as np
import pandas as pd


class Environment():
    def __init__(self, env, env_d, max_actions, max_emptyflights, init_fault,
                 df_fault, df_limit, df_close, df_flytime, base_date, df_plane_type, df_first, df_last, df_special_passtime):
        self.default_env = env
        self.env = self.default_env.copy()
        self.default_fault = init_fault
        self.fault = self.default_fault

        # 环境维度
        self.env_d = env_d
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
        # 飞机-类型表
        self.df_plane_type = df_plane_type
        # 边界表-最早起飞
        self.df_first = df_first
        # 边界表-最晚起飞
        self.df_last = df_last
        # 特殊过站时间表(默认就小于50分钟过站时间的航班)
        self.df_special_passtime = df_special_passtime

        # 最大提前时间(仅限国内)
        self.time_diff_e = -6 * 60
        # 最大国内提前时间 24 * 60
        self.time_diff_l_1 = 24 * 60
        # 最大国际提前时间 36 * 60
        self.time_diff_l_0 = 36 * 60

        # Loss 计数及重要系数
        # 0, 100000:违反硬约束，失效/故障/台风等
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
        elif action_type == 2:
            return_count = self.do_action_flightchange(action[0], action[6], action[4])

        if return_count == -1:
            # 触发了立即退出的硬约束
            end = True
        else:
            # 操作计数+1
            self.action_count += return_count
            # 达到最大操作数限制时结束
            if self.action_count >= self.max_actions:
                end = True

        return self.env, self.action_log, self.action_count, end

    # 硬约束检测函数
    # checktype: 0航站衔接、1航线-飞机限制、2机场关闭、3过站时间、4故障/台风、5边界禁止-最早、6边界禁止-最晚
    # 7边界机场一致性约束-最早起飞机场、8边界机场一致性约束-最晚降落机场
    def check_hard_constraint(self, row1, row2 = np.zeros([0]), checktype = 0):
        have_hard_constraint = False
        # 航站衔接
        if checktype == 0:
            # 前一个航班的到达机场不等于后一个航班的起飞机场(并且不是最晚的边界航班)
            if (row1[5] != row2[4]) & (row1[66] != 1):
                # 航站衔接有问题，如果之前已经航站衔接fault那么不加，否则fault+1
                if row1[58] == 0:
                    self.fault += 1
                    self.loss_val[0] = self.fault
                    row1[58] = 1
                have_hard_constraint = True
            else:
                # 航站衔接正常了，那么检测之前是否不正常，如果由之前的不正常变成正常了，fault需要-1
                if row1[58] == 1:
                    self.fault -= 1
                    self.loss_val[0] = self.fault
                    row1[58] = 0
        # 航线飞机限制
        elif checktype == 1:
            limit_ = row1[68:]
            if row1[10] in limit_:
                # 航线飞机限制，如果之前已经fault那么不加，否则fault+1
                if row1[59] == 0:
                    self.fault += 1
                    self.loss_val[0] = self.fault
                    row1[59] = 1
                have_hard_constraint = True
            else:
                # 航线飞机限制正常了，那么检测之前是否不正常，如果由之前的不正常变成正常了，fault需要-1
                if row1[59] == 1:
                    self.fault -= 1
                    self.loss_val[0] = self.fault
                    row1[59] = 0
        # 机场关闭
        elif checktype == 2:
            time_d_0 = row1[7]  # 起飞时间，0点为基准
            time_a_0 = row1[9]  # 到达时间，0点为基准
            # 机场关闭硬约束标志
            row1[60] = 0
            # 起飞机场
            if (time_d_0 > row1[33]) & (time_d_0 < row1[34]):
                if row1[37] == 0:
                    self.fault += 1
                    self.loss_val[0] = self.fault
                    row1[37] = 1
                have_hard_constraint = True
                # 机场关闭硬约束标志
                row1[60] = 1
            else:
                if row1[37] == 1:
                    self.fault -= 1
                    self.loss_val[0] = self.fault
                    row1[37] = 0
            # 降落机场
            if (time_a_0 > row1[38]) & (time_a_0 < row1[39]):
                if row1[42] == 0:
                    self.fault += 1
                    self.loss_val[0] = self.fault
                    row1[42] = 1
                have_hard_constraint = True
                # 机场关闭硬约束标志
                row1[60] = 1
            else:
                if row1[42] == 1:
                    self.fault -= 1
                    self.loss_val[0] = self.fault
                    row1[42] = 0
        # 过站时间
        elif checktype == 3:
            # 过站时间小于50分钟并且不为最晚的边界航班
            if (row1[45] < 50) & (row1[66] != 1):
                # 小于50分钟的话检测是否位于特殊过站时间表内
                r_ = self.df_special_passtime[(self.df_special_passtime['航班ID'] == row1[0])
                                              & (self.df_special_passtime['后继航班ID'] == row1[44])
                                              & (row1[45] >= self.df_special_passtime['过站时间'])]
                # 符合特殊过站时间(不小于原先的过站时间)
                if len(r_) > 0:
                    # 过站时间正常了，之前如果为不正常，那么fault-1
                    if row1[61] == 1:
                        self.fault -= 1
                        self.loss_val[0] = self.fault
                        row1[61] = 0
                else:
                    if row1[61] == 0:
                        self.fault += 1
                        self.loss_val[0] = self.fault
                        row1[61] = 1
                    have_hard_constraint = True
            else:
                # 过站时间正常了，之前如果为不正常，那么fault-1
                if row1[61] == 1:
                    self.fault -= 1
                    self.loss_val[0] = self.fault
                    row1[61] = 0
        # 故障/台风
        elif checktype == 4:
            time_d = row1[6]  # 起飞时间
            time_a = row1[8]  # 到达时间
            next_time_d = row2[6]   # 后一个航班的起飞时间
            row1[62] = 0
            # 起飞故障
            if (time_d > row1[14]) & (time_d < row1[15]):
                if row1[13] == 0:
                    self.fault += 1
                    self.loss_val[0] = self.fault
                    row1[13] = 1
                have_hard_constraint = True
                row1[62] = 1
            else:
                if row1[13] == 1:
                    self.fault -= 1
                    self.loss_val[0] = self.fault
                    row1[13] = 0

            # 降落故障
            if (time_a > row1[17]) & (time_a < row1[18]):
                if row1[16] == 0:
                    self.fault += 1
                    self.loss_val[0] = self.fault
                    row1[16] = 1
                have_hard_constraint = True
                row1[62] = 1
            else:
                if row1[16] == 1:
                    self.fault -= 1
                    self.loss_val[0] = self.fault
                    row1[16] = 0

            # 停机故障
            # 新增的((time_a < row1[32]) & (time_a > row1[31]) & (next_time_d == 0))，
            # 用于无后续航班时的验证(原先有后继航班时有停机故障，去掉后继航班后可能没有停机故障了)
            if ((time_a < row1[32]) & (next_time_d > row1[31])) \
                    | ((time_a < row1[32]) & (time_a > row1[31]) & (next_time_d == 0)):
                if row1[29] == 0:
                    self.fault += 1
                    self.loss_val[0] = self.fault
                    row1[29] = 1
                have_hard_constraint = True
                row1[62] = 1
            else:
                if row1[29] == 1:
                    self.fault -= 1
                    self.loss_val[0] = self.fault
                    row1[29] = 0
        # 边界禁止-最早
        elif checktype == 5:
            # 边界约束-最早
            if row1[65] == 1:
                if row1[63] == 0:
                    self.fault += 1
                    self.loss_val[0] = self.fault
                    row1[63] = 1
                have_hard_constraint = True
            else:
                if row1[63] == 1:
                    self.fault -= 1
                    self.loss_val[0] = self.fault
                    row1[63] = 0
        # 边界禁止-最晚
        elif checktype == 6:
            # 边界约束-最晚
            if row1[66] == 1:
                if row1[64] == 0:
                    self.fault += 1
                    self.loss_val[0] = self.fault
                    row1[64] = 1
                have_hard_constraint = True
            else:
                if row1[64] == 1:
                    self.fault -= 1
                    self.loss_val[0] = self.fault
                    row1[64] = 0
        # 边界机场一致性约束-最早起飞机场
        elif checktype == 7:
            # 最早的边界航班飞机起飞机场必须与该飞机最初环境中最早起飞机场一致
            plane_id = row1[10]
            airport_d = row1[4]
            airport_old_d = 0
            r_ = self.df_first[self.df_first['飞机ID'] == plane_id]
            if len(r_) > 0:
                airport_old_d = r_.iloc[0][3]
            # 机场不一致，违反边界机场一致性约束
            if airport_d != airport_old_d:
                if row1[63] == 0:
                    self.fault += 1
                    self.loss_val[0] = self.fault
                    row1[63] = 1
                have_hard_constraint = True
            else:
                if row1[63] == 1:
                    self.fault -= 1
                    self.loss_val[0] = self.fault
                    row1[63] = 0
        # 边界机场一致性约束-最晚降落机场
        elif checktype == 8:
            # 最晚的边界航班飞机降落机场必须与该飞机最初环境中最晚降落机场一致
            plane_id = row1[10] # 飞机ID
            airport_d = row1[5] # 降落机场
            airport_old_d = 0 # 最初最晚降落机场
            r_ = self.df_last[self.df_last['飞机ID'] == plane_id]
            if len(r_) > 0:
                airport_old_d = r_.iloc[0][4]
            # 机场不一致，违反边界机场一致性约束
            if airport_d != airport_old_d:
                if row1[64] == 0:
                    self.fault += 1
                    self.loss_val[0] = self.fault
                    row1[64] = 1
                have_hard_constraint = True
            else:
                if row1[64] == 1:
                    self.fault -= 1
                    self.loss_val[0] = self.fault
                    row1[64] = 0

        return have_hard_constraint


    # 动作处理：取消航班
    # lineID: 需要取消的航班ID
    # 本航班取消，动作日志表添加记录
    # 更新先导后继航班
    # 更新过站时间(取消航班只会让过站时间增加)
    # 暂时保留的硬约束：
    #   取消之后检查先后航站衔接是否发生问题，增加fault计数(默认取消一个航班前后衔接必然发生问题，但是连续取消多个航班可能衔接又正常了)
    # 直接退出的硬约束：
    #   边界禁止调整
    # 不存在的硬约束:
    #   机场关闭、过站时间、故障/台风、航线-飞机限制
    # 不退出也不处理的情况：
    #   已经取消过
    def do_action_cancel(self, lineID):
        # 处理的航班
        row = self.env[lineID - 1]

        # 检测是否边界约束-最早
        f_ = self.check_hard_constraint(row, checktype=5)
        # 检测是否边界约束-最晚
        l_ = self.check_hard_constraint(row, checktype=6)

        if f_ | l_:
            return -1
        else:
            if row[52] == 1:
                # 已经取消过的不处理，直接退出
                return 0
            else:
                # 取消标记
                row[52] = 1
                # ***因为取消航班有边界约束，所以能够被取消的航班必定有先导和后继
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

                ########################################################################################################
                # 硬约束检测
                # 航站衔接
                self.check_hard_constraint(row1=row_pre, row2=row_next, checktype=0)

                return 1

    # 动作处理：换飞机
    # lineID: 需要换飞机的航班ID
    # new_planeID: 新飞机ID
    # time_d: 调整后的起飞时间，飞行时间与原飞行时间相同，必须是int型，与基准日期的差异分钟数
    # 原先先导后继航班重算
    # 更新过站时间
    # 插入到新的航班序列中，重算先导后继
    # 暂时保留的硬约束：
    #   航站衔接、机场关闭、过站时间、故障/台风、边界禁止
    # 直接退出的硬约束：
    #   航线-飞机限制、提前、延误时间限制
    # 不存在的硬约束:
    #   NA
    # 不退出也不处理的情况：
    #   已经换过飞机、飞机ID一样、提前仅限国内航班
    def do_action_flightchange(self, lineID, new_planeID, time_d = 0):
        # 处理的航班
        row = self.env[lineID - 1]
        # 已经换过飞机或者飞机ID一样的不处理但是也不结束
        # 起飞时间提前仅限国内航班，国际航班提前则直接退出不处理(国际航班类型=0)
        if (row[53] == 1) | (row[53] == 2) | (row[10] == new_planeID) | (time_d !=0 & time_d < row[6] & row[2] == 0):
            return 0
        else:
            ############################################################################################################
            # 基本环境信息更新
            # 获取飞机类型
            plane_type = row[11]
            r_ = self.df_plane_type[self.df_plane_type['飞机ID'] == new_planeID]
            if len(r_) > 0:
                plane_type = r_.iloc[0][1]
            else:
                return -1
            # 设置换飞机状态标记
            if row[11] != plane_type:
                row[53] = 2
            else:
                row[53] = 1
            # 更新起降时间
            # 飞行时间
            flytime = row[8] - row[6]
            # 由于系统用了两个起降时间，分别是与基准日期的差值和与当天0天的差值，分别计算这两组时间的差异
            basetime_diff_d = row[6] - row[7]
            basetime_diff_a = row[8] - row[9]
            # 新的起飞时间
            row[6] = time_d
            row[7] = row[6] - basetime_diff_d
            # 新的降落时间
            row[8] = time_d + flytime
            row[9] = row[8] - basetime_diff_a
            # 更新飞机ID和类型
            row[10] = new_planeID
            row[11] = plane_type

            ############################################################################################################
            # 直接退出的硬约束检测
            # 如果是航线-飞机限制、提前、延误时间限制则直接结束
            # 新起飞时间与原起飞时间之差
            time_diff = 0
            if time_d != 0:
                time_diff = time_d - row[6]

            # 提前最多6小时(仅限国内航班)
            time_diff_e = self.time_diff_e
            # 延误最多24小时(国内)36小时(国际)
            time_diff_l = self.time_diff_l_1
            if row[2] == 0:
                time_diff_l = self.time_diff_l_0

            if self.check_hard_constraint(row, checktype=1) | (time_diff > time_diff_e) | (time_diff < time_diff_l):
                return -1
            else:
                #######################################################################################################
                # 先获取本航班的先导后继航班(类似于取消航班)
                # 原先的先导航班
                old_id_pre = row[43]
                # 原先的后继航班
                old_id_next = row[44]
                # 原先的先导后继航班都存在的情况下
                if (old_id_pre != 0) & (old_id_next != 0):
                    # 原先的先导航班
                    old_row_pre = self.env[old_id_pre - 1]
                    # 原先的后继航班
                    old_row_next = self.env[old_id_next - 1]
                    # 更新原先的先导后继航班的关联(即跨过本航班之后直接连接起来，与航班取消类似)
                    # 先导航班的后继航班变成本航班的后继航班
                    old_row_pre[44] = old_row_next[0]
                    # 后继航班的先导变成本航班的先导航班
                    old_row_next[43] = old_row_pre[0]
                    # 过站时间更新
                    old_row_pre[45] = old_row_next[6] - old_row_pre[8]
                    ####################################################################################################
                    # 硬约束检测(针对原先的先导航班)
                    # 本步骤不会变化的硬约束：机场关闭、边界约束
                    # 本步骤需要检测的硬约束：航站衔接、过站时间、故障/台风(停机)
                    # 航站衔接
                    self.check_hard_constraint(row1=old_row_pre, row2=old_row_next, checktype=0)
                    # 过站时间
                    self.check_hard_constraint(row1=old_row_pre, row2=old_row_next, checktype=3)
                    # 故障/台风(停机)
                    self.check_hard_constraint(row1=old_row_pre, row2=old_row_next, checktype=4)

                    # 硬约束检测(针对原先的后继航班)
                    # 本步骤不会变化的硬约束：机场关闭、故障/台风、航站衔接、过站时间、边界约束
                    # 本步骤需要检测的硬约束：无

                    # 硬约束检测结束
                    ####################################################################################################

                elif old_id_pre != 0:
                    # 仅有先导航班，本航班之前为边际航班-最晚，原先导航班变为了边际航班-最晚
                    # 原先的先导航班
                    old_row_pre = self.env[old_id_pre - 1]
                    # 原先导航班的后继航班=0
                    old_row_pre[44] = 0
                    # 原先导航班过站时间 = 0
                    old_row_pre[45] = 0
                    # 原先导航班边际状态设置-最晚
                    old_row_pre[66] = 1
                    ####################################################################################################
                    # 硬约束检测(针对原先的先导航班)
                    # 本步骤不会变化的硬约束：机场关闭
                    # 本步骤需要检测的硬约束：航站衔接、过站时间、边界约束、故障/台风(停机)
                    # 航站衔接
                    self.check_hard_constraint(row1=old_row_pre, row2=np.zeros([self.env_d]), checktype=0)
                    # 过站时间
                    self.check_hard_constraint(row1=old_row_pre, row2=np.zeros([self.env_d]), checktype=3)
                    # 边界机场一致性约束-最晚降落机场
                    self.check_hard_constraint(row1=old_row_pre, checktype=8)
                    # 故障/台风(停机)
                    self.check_hard_constraint(row1=old_row_pre, row2=np.zeros([self.env_d]), checktype=4)
                    # 硬约束检测结束
                    ####################################################################################################

                elif old_id_next !=0:
                    # 仅有后继航班，本航班之前为边际航班-最早，原后继航班变成了边际航班-最早
                    # 原先的后继航班
                    old_row_next = self.env[old_id_next - 1]
                    # 原后继航班的先导航班=0
                    old_row_next[43] = 0
                    # 原先导航班边际状态设置-最早
                    old_row_next[65] = 1
                    ####################################################################################################
                    # 硬约束检测(针对原先的后继航班)
                    # 本步骤不会变化的硬约束：机场关闭、故障/台风、航站衔接、过站时间
                    # 本步骤需要检测的硬约束：边界约束
                    # 边界机场一致性约束-最晚降落机场
                    self.check_hard_constraint(row1=old_row_next, checktype=7)
                    # 硬约束检测结束
                    ####################################################################################################

                else:
                    # 都空，不可能吧
                    return -1

                #######################################################################################################
                # 处理本航班，相当于在其他飞机的航班链条上插入本航班
                # 先将本航班的先导后继以及边际标志都清空
                row[43] = row[44] = row[45] = 0
                row[63] = row[64] = row[65] = row[66] = 0

                df = pd.DataFrame(self.env)
                # 获取先导航班(起飞时间小于本航班起飞时间，起飞时间从大到小排列的第一条记录
                r_ = df[(df[10] == new_planeID) & (df[6] < time_d)].sort_values(by=6, ascending=False)

                if len(r_) > 0:
                    ####################################################################################################
                    # 如果有先导航班
                    new_id_pre = r_.iloc[0][0]
                    new_row_pre = self.env[new_id_pre - 1]
                    # 后继航班ID
                    new_id_next = new_row_pre[44]
                    # 先导航班的后继航班为本航班
                    new_row_pre[44] = lineID
                    # 本航班的先导航班
                    row[43] = new_id_pre
                    # 更新先导航班的过站时间
                    pass_time = row[6] - new_row_pre[8]
                    # 更新过站时间
                    new_row_pre[45] = pass_time

                    ####################################################################################################
                    # 硬约束检测(针对先导航班检测)
                    # 本步骤不会变化的硬约束：机场关闭
                    # 本步骤需要检测的硬约束：故障/台风(停机)、航站衔接、过站时间

                    # 故障/台风(停机)
                    self.check_hard_constraint(row1=new_row_pre, row2=row, checktype=4)
                    # 航站衔接
                    self.check_hard_constraint(row1=new_row_pre, row2=row, checktype=0)
                    # 过站时间
                    self.check_hard_constraint(row1=new_row_pre, checktype=3)

                    # 硬约束检测结束
                    ####################################################################################################

                    # 是否存在后继航班
                    if new_id_next > 0:
                        new_row_next = self.env[new_id_next - 1]
                        # 更新本航班的后继航班
                        row[44] = new_id_next
                        # 后继航班的先导航班就是本航班
                        new_row_next[43] = lineID
                        # 更新过站时间
                        pass_time = new_row_next[6] - row[8]
                        row[45] = pass_time
                        ################################################################################################
                        # 硬约束检测(针对本航班检测)
                        # 本步骤不会变化的硬约束：
                        # 本步骤需要检测的硬约束：故障/台风(停机)、航站衔接、过站时间、机场关闭

                        # 故障/台风(停机)
                        self.check_hard_constraint(row1=row, row2=new_row_next, checktype=4)
                        # 航站衔接
                        self.check_hard_constraint(row1=row, row2=new_row_next, checktype=0)
                        # 过站时间
                        self.check_hard_constraint(row1=row, checktype=3)
                        # 机场关闭
                        self.check_hard_constraint(row1=row, checktype=2)

                        # 硬约束检测结束
                        ################################################################################################

                    else:
                        # 不存在后继航班，那么本航班就成为了边际航班-最晚(也就是说原先那个先导航班就是原边际航班，现在被取代之)
                        # 这时需要检查边际约束-最晚，到达机场必须和原先本飞机最晚到达机场一致
                        # 更新后继航班=0
                        row[44] = 0
                        # 更新是否是边际航班-最晚
                        row[66] = 1
                        # 先导航班不再是边际航班了
                        new_row_pre[66] = 0

                        ################################################################################################
                        # 硬约束检测(针对本航班检测)
                        # 本步骤不会变化的硬约束：航站衔接、过站时间
                        # 本步骤需要检测的硬约束：故障/台风、机场关闭、边界机场一致性约束-最晚降落机场

                        # 故障/台风
                        self.check_hard_constraint(row1=row, row2=np.zeros([self.env_d]), checktype=4)
                        # 机场关闭
                        self.check_hard_constraint(row1=row, checktype=2)
                        # 边界机场一致性约束-最晚降落机场
                        self.check_hard_constraint(row1=row, checktype=8)

                        # 硬约束检测(针对先导航班检测)
                        # 先导航班由边际航班-最晚 变成了非边际航班，需要重算边际控制
                        self.check_hard_constraint(row1=new_row_pre, checktype=6)

                        # 硬约束检测结束
                        ################################################################################################

                        #########有先导航班情况结束#########
                else:
                    # 没有先导航班，那么本航班就成为了边际航班-最早
                    # 获取后继航班
                    r_ = df[(df[10] == new_planeID) & (df[6] > time_d)].sort_values(by=6, ascending=True)
                    # 如果有后继航班
                    if len(r_) > 0:
                        new_id_next = r_.iloc[0][0]
                        new_row_next = self.env[new_id_next -1]

                        # 更新本航班的后继航班
                        row[44] = new_id_next
                        # 后继航班的先导航班就是本航班
                        new_row_next[43] = lineID
                        # 更新过站时间
                        pass_time = new_row_next[6] - row[8]
                        row[45] = pass_time

                        # 更新先导航班=0
                        row[43] = 0
                        # 更新是否是边际航班-最早
                        row[65] = 1
                        # 后继航班不再是边际航班了-最早
                        new_row_next[65] = 0

                        ################################################################################################
                        # 硬约束检测(针对本航班检测)
                        # 本步骤不会变化的硬约束：
                        # 本步骤需要检测的硬约束：航站衔接、过站时间、故障/台风、机场关闭、边界机场一致性约束-最早起飞机场

                        # 故障/台风(停机)
                        self.check_hard_constraint(row1=row, row2=new_row_next, checktype=4)
                        # 航站衔接
                        self.check_hard_constraint(row1=row, row2=new_row_next, checktype=0)
                        # 过站时间
                        self.check_hard_constraint(row1=row, checktype=3)
                        # 机场关闭
                        self.check_hard_constraint(row1=row, checktype=2)
                        # 边界机场一致性约束-最早机场
                        self.check_hard_constraint(row1=row, checktype=7)

                        # 硬约束检测(针对后继航班检测)
                        # 后继航班由边际航班-最早 变成了非边际航班，需要重算边际控制
                        self.check_hard_constraint(row1=new_row_next, checktype=5)

                        # 硬约束检测结束
                        ################################################################################################

                        #########没有先导航班情况结束#########

                    else:
                        # 不应该没有后继航班
                        return -1

                # 换飞机航班数+1*重要系数 (只有机型发生变化才会记录loss)
                if row[53] == 2:
                    self.loss_val[3] += 1 * row[12]
                # action 日志更新(无论机型是否变化都要记录)
                log = self.action_log[self.action_count]
                # 航班ID
                log[0] = lineID
                # 起飞时间
                log[3] = row[6]
                # 降落时间
                log[4] = row[8]
                # 飞机ID
                log[5] = new_planeID
                return 1


    # 动作处理：调机(新增一个空飞航班，仅针对国内航班)
    # airport_d: 起飞机场
    # airport_a: 到达机场
    # time_d: 起飞时间
    # time_a: 到达时间
    # planeID: 飞机ID
    # 在原先飞机序列上插入这个新航班
    # 重算先导后继，过站时间
    # 暂时保留的硬约束：
    #   航站衔接、机场关闭、过站时间、故障/台风、边界禁止
    # 直接退出的硬约束：
    #   航线-飞机限制
    # 不存在的硬约束:
    #   NA
    # 不退出也不处理的情况：
    #   调机仅限国内航班
    def do_action_emptyflights(self, airport_d, airport_a, time_d, time_a, planeID):
        return 1

    # 调整时间
    # lineID: 航班ID
    # time_d: 调整后的起飞时间
    # 调整起飞降落时间
    # 不重算先导后继
    # 重算硬约束
    # 暂时保留的硬约束：
    #   机场关闭、过站时间、故障/台风
    # 直接退出的硬约束：
    #   提前、延误时间限制
    # 不存在的硬约束:
    #   航线-飞机限制、航站衔接、边界禁止
    # 不退出也不处理的情况：
    #   提前仅限国内航班
    def do_action_changetime(self, lineID, time_d):
        return 1

    # 联程拉直
    # lineID: 航班ID
    # time_d: 调整后的起飞时间
    # 调整起飞降落机场、断开与另外一段联程航班关联，
    # 再取消联程中另外一段航班
    # 先导后继不重算(使用原先完整的联程航班的先导后继)，重算过站时间
    # 重算硬约束
    # 暂时保留的硬约束：
    #   机场关闭、过站时间、故障/台风
    # 直接退出的硬约束：
    #   提前、延误时间限制
    # 不存在的硬约束:
    #   航线-飞机限制、航站衔接、边界禁止
    # 不退出也不处理的情况：
    #   拉直仅限两段都是国内航班，仅限中间机场发生故障时方可拉直
    def do_action_flightstraighten(self, lineID, time_d):
        return 1


    def show(self):
        print('loss_val: ', self.loss_val)
        print('fault: ', self.fault)
        print('action_log: ', self.action_log)
        print('action_count: ', self.action_count)

