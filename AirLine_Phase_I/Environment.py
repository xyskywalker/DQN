import numpy as np
import pandas as pd
import datetime


class Environment():
    def __init__(self, env, max_actions, max_emptyflights, init_fault,
                 df_fault, df_limit, df_close, df_flytime, base_date, df_plane_type, df_first, df_last, df_special_passtime,
                 domestic_airport):
        self.default_env = env
        self.env = self.default_env.copy()
        self.row_count = len(self.env)
        self.default_fault = init_fault
        self.fault = self.default_fault

        # 环境维度
        self.env_d = self.env.shape[1]
        # 故障表
        self.df_fault = df_fault
        # 航线-飞机限制表
        self.df_limit = df_limit
        # 机场关闭表
        self.df_close = df_close
        # 飞行时间表
        self.df_flytime = df_flytime
        # 基准日期
        self.base_date = base_date.to_pydatetime()
        # 飞机-类型表
        self.df_plane_type = df_plane_type
        # 边界表-最早起飞
        self.df_first = df_first
        # 边界表-最晚起飞
        self.df_last = df_last
        # 特殊过站时间表(默认就小于50分钟过站时间的航班)
        self.df_special_passtime = df_special_passtime
        # 国内机场列表
        self.domestic_airport = domestic_airport


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
        self.loss_val = np.zeros([7], dtype=np.float32)
        self.loss_val[0] = self.fault

        # 最大允许调机的数量
        self.max_emptyflights = max_emptyflights
        self.max_emptyflights_count = 0

        # 环境中增加空的位置用于容纳可能的调机航班
        tempArr = np.zeros([self.max_emptyflights, self.env_d], dtype=np.int32)
        self.env = np.row_stack((self.env, tempArr))

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
        self.action_log = np.zeros([self.max_actions, 9], dtype=np.int32)
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

    # 通过相对于基准日期的分钟数获取相对于当天0点的分钟数
    def get_minutes_0(self, minutes_basedate):
        dt = self.base_date + datetime.timedelta(minutes=float(minutes_basedate))
        return dt.hour * 60 + dt.minute

    # action
    # 0: Line ID
    # 1: Action Type: 0调机，1取消，2换飞机，3调整时间，4联程拉直
    # 2: 起飞机场
    # 3: 降落机场
    # 4: 起飞时间
    # 5: 飞机ID
    def step(self, action):
        # 结束标识
        end = False

        # 达到最大操作数限制时结束
        if self.action_count >= self.max_actions - 2:
            end = True
        else:
            # 处理不同的action type
            action_type = action[1]
            return_count = 0
            # 调机
            if action_type == 0:
                return_count = self.do_action_emptyflights(airport_d=action[2]
                                                           , airport_a=action[3]
                                                           , time_d=action[4]
                                                           , planeID=action[5])
            # 取消
            elif action_type == 1:
                return_count = self.do_action_cancel(lineID=action[0])
            # 换飞机
            elif action_type == 2:
                return_count = self.do_action_flightchange(lineID=action[0], new_planeID=action[5], time_d=action[4])
            # 调整时间
            elif action_type == 3:
                return_count = self.do_action_changetime(lineID=action[0], time_d=action[4])
            # 联程拉直
            elif action_type == 4:
                return_count = self.do_action_flightstraighten(lineID=action[0], time_d=action[4])

            if return_count == -1:
                # 触发了立即退出的硬约束
                end = True
            else:
                # 操作计数+1
                self.action_count += return_count

        return self.env, self.action_log, self.action_count, end

    # 硬约束检测函数
    # checktype: 0航站衔接、1航线-飞机限制、2机场关闭、3过站时间、4故障/台风、5边界禁止-最早、6边界禁止-最晚
    # 7边界机场一致性约束-最早起飞机场、8边界机场一致性约束-最晚降落机场
    def check_hard_constraint(self, row1, row2 = np.zeros([0]), checktype = 0):
        have_hard_constraint = False
        # 0 航站衔接
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

        # 1 航线飞机限制
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

        # 2 机场关闭
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

        # 3 过站时间
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

        # 4 故障/台风
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

        # 5 边界禁止-最早
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

        # 6 边界禁止-最晚
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

        # 7 边界机场一致性约束-最早起飞机场
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

        # 8 边界机场一致性约束-最晚降落机场
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
    #   已经取消过、做过其他调整的:换飞机、调整时间
    def do_action_cancel(self, lineID):
        # 处理的航班
        row = self.env[lineID - 1]

        # 检测是否边界约束-最早
        f_ = self.check_hard_constraint(row, checktype=5)
        # 检测是否边界约束-最晚(联程拉直导致的第二段取消时不检查是否是最晚的边际航班)
        l_ = self.check_hard_constraint(row, checktype=6) & (row[55] == 0)

        if f_ | l_:
            return -1
        else:
            # 不退出也不处理的情况：
            #   已经取消过
            #   做过其他调整的:换飞机、调整时间，由于联程拉直需要取消的第二段无论之前有没有调整过都要取消
            if ((row[52] == 1) | (row[53] != 0) | (row[54] != 0)) & (row[55] == 0):
                return 1
            else:
                # 取消标记
                row[52] = 1
                # ***因为取消航班有边界约束，所以能够被取消的航班必定有先导和后继（联程拉直导致取消第二段除外）
                # 先导航班：
                row_pre = self.env[row[43] - 1]
                # 后继航班存在的情况下处理(联程拉直导致的第二段取消也不做任何处理)：
                if (row[44] > 0) & (row[55] == 0):
                    row_next = self.env[row[44] - 1]
                    # 先导航班的后继航班变成本航班的后继航班
                    row_pre[44] = row_next[0]
                    # 后继航班的先导变成本航班的先导航班
                    row_next[43] = row_pre[0]
                    # 过站时间更新
                    row_pre[45] = row_next[6] - row_pre[8]
                    ####################################################################################################
                    # 硬约束检测
                    # 航站衔接
                    self.check_hard_constraint(row1=row_pre, row2=row_next, checktype=0)

                # action 日志更新
                log = self.action_log[self.action_count]
                log[0] = lineID
                log[6] = 1
                # 联程拉直标记
                if row[55] == 1:
                    log[7] = 1
                else:
                    # 取消航班数+1*重要系数(联程拉直不做取消统计)
                    self.loss_val[2] += 1 * row[12]

                return 1

    # 动作处理：换飞机
    # lineID: 需要换飞机的航班ID
    # new_planeID: 新飞机ID
    # time_d: 起飞时间调整量，int型，范围 -6 * 60 ~ 36 * 60
    # 原先先导后继航班重算
    # 更新过站时间
    # 插入到新的航班序列中，重算先导后继
    # 联程航班有特殊性：
    #   lineID自动变成第一段的ID、调整时间也针对第一段、两段都换成同一架飞机、前后两段的关联不能断开
    # 暂时保留的硬约束：
    #   航站衔接、机场关闭、过站时间、故障/台风、边界禁止、联程航班前后飞机必须一致
    # 直接退出的硬约束：
    #   航线-飞机限制、提前、延误时间限制
    # 不存在的硬约束:
    #   NA
    # 不退出也不处理的情况：
    #   已经取消的航班、已经换过飞机、飞机ID一样、提前仅限国内航班
    def do_action_flightchange(self, lineID, new_planeID, time_d = 0):
        # 处理的航班
        row = self.env[lineID - 1]
        ############################################################################################################
        # 联程航班的特殊处理
        # 是否联程航班
        is_connecting_flight = False
        # 默认联程航班为本航班后继航班
        connecting_flight_row = self.env[row[44] - 1]
        # 保留原先的lineID
        lineID_old = lineID
        if row[46] == 1:
            is_connecting_flight = True
            if row[47] == row[43]:
                # 如果联程航班是本航班的先导航班，那么lineID移动到先导航班
                lineID = row[43]
                connecting_flight_row = row
                row = self.env[lineID - 1]
        #
        ############################################################################################################
        # 已经换过飞机或者飞机ID一样的不处理但是也不结束
        # 已经取消的航班、已经取消的、改变过时间的
        # 起飞时间提前仅限国内航班，国际航班提前则直接退出不处理(国际航班类型=0)
        if (row[52] == 1) | (row[53] == 1) | (row[53] == 2) | (row[10] == new_planeID) | ((time_d < 0) & (row[2] == 0)):
            return 1
        else:
            ############################################################################################################
            # 基本环境信息更新
            # 获取飞机类型
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
            # 新的起飞时间
            row[6] += time_d
            row[7] = self.get_minutes_0(row[6])
            # 新的降落时间
            row[8] += time_d + flytime
            row[9] = self.get_minutes_0(row[8])
            # 更新飞机ID和类型
            row[10] = new_planeID
            row[11] = plane_type

            ######更新联程航班另一段的飞机信息(起降时间暂不修改)######
            if is_connecting_flight:
                # 设置换飞机状态标记
                if connecting_flight_row[11] != plane_type:
                    connecting_flight_row[53] = 2
                else:
                    connecting_flight_row[53] = 1
                # 更新飞机ID和类型
                connecting_flight_row[10] = new_planeID
                connecting_flight_row[11] = plane_type
                # 更新过站时间
                row[45] = connecting_flight_row[6] = row[8]
                ########################################################################################################
                # 联程航班的特殊硬约束检测
                # 过站时间硬约束检测
                self.check_hard_constraint(row1=row, row2=connecting_flight_row, checktype=3)
                # 故障/台风(停机)
                self.check_hard_constraint(row1=row, row2=connecting_flight_row, checktype=4)
                # 机场关闭
                self.check_hard_constraint(row1=row, checktype=2)
                ########################################################################################################

            ############################################################################################################
            # 直接退出的硬约束检测
            # 如果是航线-飞机限制、提前、延误时间限制则直接结束
            # 提前最多6小时(仅限国内航班)
            time_diff_e = self.time_diff_e
            # 延误最多24小时(国内)36小时(国际)
            time_diff_l = self.time_diff_l_1
            if row[2] == 0:
                time_diff_l = self.time_diff_l_0

            if self.check_hard_constraint(row, checktype=1) | (time_d < time_diff_e) | (time_d > time_diff_l):
                return -1
            else:
                #######################################################################################################
                # 先获取本航班的先导后继航班(类似于取消航班)
                # 原先的先导航班
                old_id_pre = row[43]
                # 原先的后继航班
                old_id_next = row[44]
                ######联程航班的后继航班ID是联程第二段的后继航班#####
                if is_connecting_flight:
                    old_id_next = connecting_flight_row[44]

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
                if is_connecting_flight is False:
                    row[43] = row[44] = row[45] = 0
                else:
                    # 联程航班第一段
                    row[43] = 0
                    # 联程航班第二段
                    connecting_flight_row[44] = connecting_flight_row[45] = 0

                row[63] = row[64] = row[65] = row[66] = 0
                connecting_flight_row[63] = connecting_flight_row[64] = connecting_flight_row[65] = connecting_flight_row[66] = 0

                df = pd.DataFrame(self.env)
                # 获取先导航班(起飞时间小于本航班起飞时间，起飞时间从大到小排列的第一条记录
                r_ = df[(df[10] == new_planeID) & (df[6] < row[6])].sort_values(by=6, ascending=False)

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
                        row_ = row
                        ######如果是联程航班，那么后继航班需要对应联程的第二段来处理
                        if is_connecting_flight:
                            row_ = connecting_flight_row
                        # 更新本航班的后继航班
                        row_[44] = new_id_next
                        # 后继航班的先导航班就是本航班
                        new_row_next[43] = row_[0]
                        # 更新过站时间
                        pass_time = new_row_next[6] - row_[8]
                        row_[45] = pass_time
                        ################################################################################################
                        # 硬约束检测(针对本航班检测)
                        # 本步骤不会变化的硬约束：
                        # 本步骤需要检测的硬约束：故障/台风(停机)、航站衔接、过站时间、机场关闭

                        # 故障/台风(停机)
                        self.check_hard_constraint(row1=row_, row2=new_row_next, checktype=4)
                        # 航站衔接
                        self.check_hard_constraint(row1=row_, row2=new_row_next, checktype=0)
                        # 过站时间
                        self.check_hard_constraint(row1=row_, checktype=3)
                        # 机场关闭
                        self.check_hard_constraint(row1=row_, checktype=2)

                        # 硬约束检测结束
                        ################################################################################################

                    else:
                        # 不存在后继航班，那么本航班就成为了边际航班-最晚(也就是说原先那个先导航班就是原边际航班，现在被取代之)
                        # 这时需要检查边际约束-最晚，到达机场必须和原先本飞机最晚到达机场一致
                        row_ = row
                        ######如果是联程航班，那么后继航班需要对应联程的第二段来处理
                        if is_connecting_flight:
                            row_ = connecting_flight_row

                        # 更新后继航班=0
                        row_[44] = 0
                        # 更新是否是边际航班-最晚
                        row_[66] = 1
                        # 先导航班不再是边际航班了
                        new_row_pre[66] = 0

                        ################################################################################################
                        # 硬约束检测(针对本航班检测)
                        # 本步骤不会变化的硬约束：航站衔接、过站时间
                        # 本步骤需要检测的硬约束：故障/台风、机场关闭、边界机场一致性约束-最晚降落机场

                        # 故障/台风
                        self.check_hard_constraint(row1=row_, row2=np.zeros([self.env_d]), checktype=4)
                        # 机场关闭
                        self.check_hard_constraint(row1=row_, checktype=2)
                        # 边界机场一致性约束-最晚降落机场
                        self.check_hard_constraint(row1=row_, checktype=8)

                        # 硬约束检测(针对先导航班检测)
                        # 先导航班由边际航班-最晚 变成了非边际航班，需要重算边际控制
                        self.check_hard_constraint(row1=new_row_pre, checktype=6)

                        # 硬约束检测结束
                        ################################################################################################

                        #########有先导航班情况结束#########
                else:
                    # 没有先导航班，那么本航班就成为了边际航班-最早
                    row_ = row
                    ######如果是联程航班，那么后继航班需要对应联程的第二段来处理
                    if is_connecting_flight:
                        row_ = connecting_flight_row

                    # 获取后继航班
                    r_ = df[(df[10] == new_planeID) & (df[6] > row_[6])].sort_values(by=6, ascending=True)
                    # 如果有后继航班
                    if len(r_) > 0:
                        new_id_next = r_.iloc[0][0]
                        new_row_next = self.env[new_id_next -1]

                        # 更新本航班的后继航班
                        row_[44] = new_id_next
                        # 后继航班的先导航班就是本航班
                        new_row_next[43] = row_[0]
                        # 更新过站时间
                        pass_time = new_row_next[6] - row_[8]
                        row_[45] = pass_time

                        ######下面两个先导和最早边际依旧处理联程中的第一段
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
                        self.check_hard_constraint(row1=row_, row2=new_row_next, checktype=4)
                        # 航站衔接
                        self.check_hard_constraint(row1=row_, row2=new_row_next, checktype=0)
                        # 过站时间
                        self.check_hard_constraint(row1=row_, checktype=3)
                        # 机场关闭
                        self.check_hard_constraint(row1=row_, checktype=2)
                        # 边界机场一致性约束-最早机场(这是依旧测试联程航班的第一段)
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
                log[0] = row[0]
                # 起飞时间
                log[3] = row[6]
                # 降落时间
                log[4] = row[8]
                # 飞机ID
                log[5] = new_planeID
                # 如果是联程航班那么要多加一条
                if is_connecting_flight:
                    # 换飞机航班数+1*重要系数 (只有机型发生变化才会记录loss)
                    if connecting_flight_row[53] == 2:
                        self.loss_val[3] += 1 * connecting_flight_row[12]
                    log = self.action_log[self.action_count]
                    # 航班ID
                    log[0] = connecting_flight_row[0]
                    # 飞机ID
                    log[5] = new_planeID
                    return 2
                else:
                    return 1

    # 动作处理：调机(新增一个空飞航班，仅针对国内航班)
    # airport_d: 起飞机场
    # airport_a: 到达机场
    # time_d: 起飞时间(与BaseDate的差值(分钟数))
    # planeID: 飞机ID
    # 在原先飞机序列上插入这个新航班
    # 重算先导后继，过站时间
    # 暂时保留的硬约束：
    #   航站衔接、机场关闭、过站时间、故障/台风
    # 直接退出的硬约束：
    #   NA
    # 不存在的硬约束:
    #   NA
    # 不退出也不处理的情况：
    #   边界禁止、航线-飞机限制、调机仅限国内航班、不在飞行时间表内
    def do_action_emptyflights(self, airport_d, airport_a, time_d, planeID):
        # 构造空行
        row_temp = np.zeros([self.env_d], dtype=np.int32)
        # 调机的航班ID
        row_temp[0] = self.row_count + self.max_emptyflights_count + 1
        # 国内
        row_temp[2] = 1
        # 起飞机场
        row_temp[4] = airport_d
        # 到底机场
        row_temp[5] = airport_a

        # 退出的条件标志
        check_ = True

        # 航线-飞机限制表构造
        arr_limit = np.array(self.df_limit[((self.df_limit['起飞机场'] == airport_d)
                                           & (self.df_limit['降落机场'] == airport_a))]['飞机ID'])

        # 起降机场必须都是国内机场
        if not ((airport_d in self.domestic_airport) & (airport_a in self.domestic_airport)):
            check_ = False

        # 获取飞机类型
        r_ = self.df_plane_type[self.df_plane_type['飞机ID'] == planeID]
        if len(r_) > 0:
            plane_type = r_.iloc[0][1]
            # 飞机ID
            row_temp[10] = planeID
            # 飞机类型
            row_temp[11] = plane_type

            if len(arr_limit) > 0:
                row_temp[68: 68 + len(arr_limit)] = arr_limit
                # 航班-飞机限制
                check_ = self.check_hard_constraint(row_temp, checktype=1) is False
        else:
            check_ = False

        # 查找飞行时间表
        r_ = self.df_flytime[(self.df_flytime['飞机机型'] == row_temp[11])
                             & (self.df_flytime['起飞机场'] == row_temp[4])
                             & (self.df_flytime['降落机场'] == row_temp[5])]
        if len(r_) > 0:
            # 飞行时间
            flytime = r_.iloc[0][3]
            # 起飞时间
            row_temp[6] = time_d
            # 降落时间
            row_temp[8] = time_d + flytime
            # 起飞时间-相对当天0点的时间
            row_temp[7] = self.get_minutes_0(row_temp[6])
            # 降落时间-相对当天0点的时间
            row_temp[9] = self.get_minutes_0(row_temp[8])
        else:
            check_ = False

        if check_:
            df = pd.DataFrame(self.env)
            # 查找先导与后继
            # 调机航班必须拥有先导与后继(边界不允许调机)
            # 获取先导航班(起飞时间小于本航班起飞时间，起飞时间从大到小排列的第一条记录
            r_p = df[(df[10] == planeID) & (df[6] < row_temp[6])].sort_values(by=6, ascending=False)
            if len(r_p) > 0:
                row_p = self.env[r_p.iloc[0][0] - 1]
                # 如果有后继航班(即先导航班的后继航班)
                if row_p[44] > 0:
                    row_n = self.env[row_p[44] - 1]

                    # 如果插在了联程航班中间
                    if (row_p[46] == 1) & (row_n[46] == 1):
                        check_ = False
                    else:
                        # 新航班的先导航班
                        row_temp[43] = row_p[0]
                        # 新航班的后继航班
                        row_temp[44] = row_n[0]
                        # 新航班的过站时间
                        row_temp[45] = row_n[6] - row_temp[8]

                        # 更新先导航班的后继与过站时间
                        row_p[44] = row_temp[0]
                        row_p[45] = row_temp[6] - row_p[8]

                        # 更新后继航班的先导
                        row_n[43] = row_temp[0]

                        ################################################################################################
                        # 附加属性处理：故障/台风；机场关闭；
                        # 13、14、15起飞机场故障(状态、开始时间、结束时间):起飞时间在范围内 & 故障类型=飞行 & 起飞机场相同
                        r_ = self.df_fault[(self.df_fault['影响类型'] == '起飞') & (airport_d == self.df_fault['机场'])]
                        if len(r_) > 0:
                            t_s = r_['开始时间'].min()
                            t_e = r_['结束时间'].max()
                            row_temp[14] = t_s
                            row_temp[15] = t_e
                            if (row_temp[6] > t_s) & (row_temp[6] < t_e):
                                row_temp[13] = 1
                                self.fault += 1
                                self.loss_val[0] = self.fault
                                # 台风场景故障状态
                                row_temp[62] = 1

                        # 16、17、18降落机场故障(状态、开始时间、结束时间):降落时间在范围内 & (故障类型=飞行|降落) & 降落机场相同
                        r_ = self.df_fault[(self.df_fault['影响类型'] == '降落') & (airport_a == self.df_fault['机场'])]
                        if len(r_) > 0:
                            t_s = r_['开始时间'].min()
                            t_e = r_['结束时间'].max()
                            row_temp[17] = t_s
                            row_temp[18] = t_e
                            if (row_temp[8] > t_s) & (row_temp[8] < t_e):
                                row_temp[16] = 1
                                self.fault += 1
                                self.loss_val[0] = self.fault
                                # 台风场景故障状态
                                row_temp[62] = 1

                        # 29、30、31、32降落机场停机限制(状态、停机限制数量、开始时间、结束时间):
                        # 时间在范围内(本航班的降落时间<结束时间 & 后继班的起飞时间>开始时间) & 故障类型=停机 & 降落机场相同
                        r_ = self.df_fault[(self.df_fault['影响类型'] == '停机') & (airport_a == self.df_fault['机场'])]
                        if len(r_) > 0:
                            self.df_fault.loc[r_.index, ['已停机数']] += 1
                            p_num = 0
                            t_s = r_['开始时间'].min()
                            t_e = r_['结束时间'].max()
                            row_temp[30] = p_num
                            row_temp[31] = t_s
                            row_temp[32] = t_e
                            if (row_temp[8] < t_e) & (row_n[6] > t_s):
                                row_temp[29] = 1
                                self.fault += 1
                                self.loss_val[0] = self.fault
                                # 台风场景故障状态
                                row_temp[62] = 1

                        # 33起飞机场关闭(相对于0点的分钟数)，34起飞机场开放(相对于0点的分钟数)，35起飞机场关闭起效日期，36起飞机场关闭失效日期，37是否起飞机场关闭
                        # 起飞机场ID一致 & 起飞时间在机场关闭的生效与失效日期之内
                        rows = np.array(self.df_close[(self.df_close['机场'] == airport_d)])
                        for row_ in rows:
                            # 关闭和开放时间之间跨越24点的处理
                            if row_[2] < row_[1]:
                                row_[2] += 24 * 60

                            row_temp[33] = row_[1]
                            row_temp[34] = row_[2]
                            row_temp[35] = row_[3]
                            row_temp[36] = row_[4]
                            if (row_temp[6] >= row_[3]) & (row_temp[6] <= row_[4]) \
                                    & (row_temp[7] > row_[1]) & (row_temp[7] < row_[2]):
                                self.fault += 1
                                self.loss_val[0] = self.fault
                                row_temp[37] = 1

                        # 38降落机场关闭(相对于0点的分钟数)，39降落机场开放(相对于0点的分钟数)，40降落机场关闭起效日期，41降落机场关闭失效日期，42是否降落机场关闭
                        # 降落机场ID一致 & 降落时间在机场关闭的生效与失效日期之内
                        rows = np.array(self.df_close[(self.df_close['机场'] == airport_a)])
                        for row_ in rows:
                            # 关闭和开放时间之间跨越24点的处理
                            if row_[2] < row_[1]:
                                row_[2] += 24 * 60

                            row_temp[38] = row_[1]
                            row_temp[39] = row_[2]
                            row_temp[40] = row_[3]
                            row_temp[41] = row_[4]
                            if (row_temp[8] >= row_[3]) & (row_temp[8] <= row_[4]) \
                                    & (row_temp[9] > row_[1]) & (row_temp[9] < row_[2]):
                                self.fault += 1
                                self.loss_val[0] = self.fault
                                row_temp[42] = 1

                        ################################################################################################
                        # 硬约束检查
                        # 先导航班的过站时间:
                        self.check_hard_constraint(row1=row_p, checktype=3)
                        # 先导航班的航站衔接
                        self.check_hard_constraint(row1=row_p, row2=row_temp, checktype=0)
                        # 先导航班的机场关闭
                        self.check_hard_constraint(row1=row_p, checktype=2)
                        # 先导航班的故障/台风
                        self.check_hard_constraint(row1=row_p, row2=row_temp, checktype=4)

                        # 新的空飞航班检测
                        # 过站时间
                        self.check_hard_constraint(row1=row_temp, checktype=3)
                        # 航站衔接
                        self.check_hard_constraint(row1=row_temp, row2=row_n, checktype=0)

                        # 后继航班无需检测
                else:
                    check_ = False
            else:
                check_ = False

        if check_:
            # 更新到环境上
            self.env[row_temp[0] - 1] = row_temp
            # 调机航班数+1
            self.loss_val[1] += 1
            # action 日志更新(无论机型是否变化都要记录)
            log = self.action_log[self.action_count]
            # 航班ID
            log[0] = row_temp[0]
            # 起飞机场
            log[1] = row_temp[4]
            # 降落机场
            log[2] = row_temp[5]
            # 起飞时间
            log[3] = row_temp[6]
            # 降落时间
            log[4] = row_temp[8]
            # 飞机ID
            log[5] = planeID

        return 1

    # 调整时间
    # lineID: 航班ID
    # time_d: 起飞时间调整量，int型，范围 -6 * 60 ~ 36 * 60
    # 调整起飞降落时间
    # 不重算先导后继
    # 重算硬约束
    # 暂时保留的硬约束：
    #   机场关闭、过站时间、故障/台风
    # 直接退出的硬约束：
    #   NA
    # 不存在的硬约束:
    #   航线-飞机限制、航站衔接、边界禁止
    # 不退出也不处理的情况：
    #   提前仅限国内航班、提前、延误时间限制
    def do_action_changetime(self, lineID, time_d):
        row = self.env[lineID - 1]
        # 提前最多6小时(仅限国内航班)
        time_diff_e = self.time_diff_e
        # 延误最多24小时(国内)36小时(国际)
        time_diff_l = self.time_diff_l_1
        if row[2] == 0:
            # 国际航班不允许提前
            time_diff_e = 0
            time_diff_l = self.time_diff_l_0

        if (time_d < time_diff_e) | (time_d > time_diff_l) | (time_d == 0):
            return 1
        else:
            # 更新起降时间
            # 重算起飞降落时间
            # 后继航班
            row_next = self.env[row[44] - 1]
            # 飞行时间
            flytime = (row[8] - row[6])
            # 新的起飞时间
            row[6] += time_d
            row[7] = self.get_minutes_0(row[6])
            # 新的降落时间
            row[8] += time_d + flytime
            row[9] = self.get_minutes_0(row[8])
            # 更新过站时间
            row[45] = row_next[6] - row[8]

            # 时间调整标记
            if time_d < 0:
                row[54] = 2
            else:
                row[54] = 1
            ############################################################################################################
            # 重算第一段的相关硬约束
            # 暂时保留的硬约束：
            #   机场关闭、过站时间、故障/台风
            # 不存在的硬约束:
            #   航线-飞机限制、航站衔接、边界禁止
            # 过站时间
            self.check_hard_constraint(row1=row, row2=row_next, checktype=3)
            # 故障/台风(停机)
            self.check_hard_constraint(row1=row, row2=row_next, checktype=4)
            # 机场关闭
            self.check_hard_constraint(row1=row, checktype=2)
            #
            ############################################################################################################

            # 调整时间(小时数)*重要系数
            self.loss_val[4] += (abs(time_d) / 60.0) * row[12]
            # action 日志更新
            log = self.action_log[self.action_count]
            # 航班ID
            log[0] = row[0]
            # 起飞时间
            log[3] = row[6]
            # 降落时间
            log[4] = row[8]

            return 1

    # 联程拉直
    # lineID: 航班ID
    # time_d: 起飞时间调整量，int型，范围 -6 * 60 ~ 36 * 60
    # 联程拉直仅限对联程航班的第一段处理
    # 调整起飞降落机场、断开与另外一段联程航班关联，
    # 再取消联程中另外一段航班
    # 先导后继不重算(使用原先完整的联程航班的先导后继)，重算过站时间
    # 重算硬约束
    # 暂时保留的硬约束：
    #   机场关闭、过站时间、故障/台风
    # 直接退出的硬约束：
    #   NA
    # 不存在的硬约束:
    #   航线-飞机限制、航站衔接、边界禁止
    # 不退出也不处理的情况：
    #   不是联程航班；提前、延误时间限制；拉直仅限两段都是国内航班，仅限中间机场发生故障时方可拉直；做过其他处理；
    def do_action_flightstraighten(self, lineID, time_d):
        # 处理的航班
        row = self.env[lineID - 1]
        connecting_flight_row = self.env[row[44] - 1]
        is_connecting_flight = False
        # 保留原先的lineID
        lineID_old = lineID
        if row[46] == 1:
            is_connecting_flight = True
            if row[47] == row[43]:
                # 如果联程航班是本航班的先导航班，那么lineID移动到先导航班
                lineID = row[43]
                connecting_flight_row = row
                row = self.env[lineID - 1]

        ################################################################################################################
        # 不退出也不处理的情况：
        #   不是联程航班；提前、延误时间限制；拉直仅限两段都是国内航班，仅限中间机场发生故障时方可拉直；做过其他处理；
        # 不是联程航班
        check_ = is_connecting_flight is False
        # 是国际航班
        check_ = check_ | ((row[2] == 0) | (connecting_flight_row[2] == 0))
        # 做过其他处理: 取消
        check_ = check_ | ((row[52] == 1) | (connecting_flight_row[52] == 1))
        # 提前、延误时间超限
        check_ = check_ | ((time_d < self.time_diff_e) | (time_d > self.time_diff_l_1))
        # 中间机场没有发生故障
        check_ = check_ | ((row[16] == 0) & (row[29] == 0)
                           & (connecting_flight_row[13] == 0) & (connecting_flight_row[25] == 0))
        if check_ is False:
            # 先导航班没有影响
            # 更新后继航班(直接连接上联程第二段的后继航班)
            row[44] = connecting_flight_row[44]
            # 降落机场=联程第二段的降落机场
            row[5] = connecting_flight_row[5]
            # 后继航班
            row_next = self.env[row[44] - 1]
            # 后继航班的先导航班变成联程中的第一段
            row_next[43] = row[0]

            # 重要系数，取两段中最大的一个
            rate = max((row[12], connecting_flight_row[12]))

            # 重算起飞降落时间
            # 飞行时间(默认取两段飞行时间之和)
            flytime = (row[8] - row[6]) + (connecting_flight_row[8] - connecting_flight_row[6])
            # 查找飞行时间表
            r_ = self.df_flytime[(self.df_flytime['飞机机型'] == row[11])
                                 & (self.df_flytime['起飞机场'] == row[4])
                                 & (self.df_flytime['降落机场'] == row[5])]
            if len(r_) > 0:
                flytime = r_.iloc[0][3]

            # 新的起飞时间
            row[6] += time_d
            row[7] = self.get_minutes_0(row[6])
            # 新的降落时间
            row[8] += time_d + flytime
            row[9] = self.get_minutes_0(row[8])
            # 更新过站时间
            row[45] = row_next[6] - row[8]

            # 航班拉直状态标记
            row[55] = 1
            connecting_flight_row[55] = 1
            # 先取消掉第二段
            self.do_action_cancel(connecting_flight_row[0])
            self.action_count += 1

            ############################################################################################################
            # 重算第一段的相关硬约束
            ## 暂时保留的硬约束：
            #   机场关闭、过站时间、故障/台风
            # 不存在的硬约束:
            #   航线-飞机限制、航站衔接、边界禁止

            # 过站时间
            self.check_hard_constraint(row1=row, row2=row_next, checktype=3)
            # 故障/台风(停机)
            self.check_hard_constraint(row1=row, row2=row_next, checktype=4)
            # 机场关闭
            self.check_hard_constraint(row1=row, checktype=2)
            #
            ############################################################################################################

            # 拉直航班数+1*重要系数
            self.loss_val[4] += 1 * rate
            # action 日志更新
            log = self.action_log[self.action_count]
            # 航班ID
            log[0] = row[0]
            # 起飞时间
            log[3] = row[6]
            # 降落时间
            log[4] = row[8]
            # 是否拉直
            log[7] = 1

            return 1
        else:
            # 直接退出
            return 1


    def show(self):
        print('loss_val: ', self.loss_val)
        print('fault: ', self.fault)
        print('action_log: ', self.action_log)
        print('action_count: ', self.action_count)

