import numpy as np
import pandas as pd


class Environment():
    def __init__(self, env, max_actions, max_emptyflights, init_fault,
                 df_fault, df_limit, df_close, df_flytime, base_date, df_plane_type, df_first, df_last):
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
        # 飞机-类型表
        self.df_plane_type = df_plane_type
        # 边界表-最早起飞
        self.df_first = df_first
        # 边界表-最晚起飞
        self.df_last = df_last

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
    def do_action_cancel(self, lineID):
        # 处理的航班
        row = self.env[lineID - 1]

        # 检测是否边界约束
        f_or_l_ = False
        if row[65] == 1:
            row[63] = 1
            f_or_l_ = True

        if row[66] == 1:
            row[64] = 1
            f_or_l_ = True

        if f_or_l_:
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
    # new_planeID: 新飞机ID
    # time_d: 调整后的起飞时间，飞行时间与原飞行时间相同，必须是int型，与基准日期的差异分钟数
    # 先导后继航班重算
    # 更新过站时间
    # 暂时保留的硬约束：
    #   航站衔接、机场关闭、过站时间、故障/台风
    # 直接退出的硬约束：
    #   航线-飞机限制
    # 不存在的硬约束:
    #   边界禁止
    def do_action_flightchange(self, lineID, new_planeID, time_d = 0):
        # 处理的航班
        row = self.env[lineID - 1]
        # 如果是航线-飞机限制则直接结束
        limit_ = row[68:]
        if new_planeID in limit_:
            # 更新硬约束标记
            row[59] = 1
            return -1
        else:
            # 已经换过飞机或者飞机ID一样的不处理
            if (row[53] == 1) | (row[53] == 2) | (row[10] == new_planeID):
                return 0
            else:
                # 获取飞机类型
                plane_type = row[11]
                r_ = self.df_plane_type[self.df_plane_type['飞机ID'] == new_planeID]
                if len(r_) > 0:
                    plane_type = r_.iloc[0][1]
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

                #######################################################################################################
                # 先处理本航班的先导后继航班(类似于取消航班)
                # 原先的先导航班
                old_id_pre = row[43]
                old_row_pre = self.env[old_id_pre - 1]
                # 原先的后继航班
                old_id_next = row[44]
                old_row_next = self.env[old_id_next - 1]
                # 更新原先的先导后继航班的关联(即跨过本航班之后直接连接起来，与航班取消类似)
                # 先导航班的后继航班变成本航班的后继航班
                old_row_pre[44] = old_row_next[0]
                # 后继航班的先导变成本航班的先导航班
                old_row_next[43] = old_row_pre[0]
                # 过站时间更新
                old_row_pre[45] = old_row_next[6] - old_row_pre[8]
                # 检测航站衔接
                if old_row_pre[5] != old_row_next[4]:
                    # 航站衔接有问题，如果之前已经航站衔接fault那么不加，否则fault+1
                    if old_row_pre[58] == 0:
                        self.fault += 1
                        self.loss_val[0] = self.fault
                        old_row_pre[58] = 1
                else:
                    # 航站衔接正常了，那么检测之前是否不正常，如果由之前的不正常变成正常了，fault需要-1
                    if old_row_pre[58] == 1:
                        self.fault -= 1
                        self.loss_val[0] = self.fault
                        old_row_pre[58] = 0

                #######################################################################################################
                # 再处理本航班，相当于在其他飞机的航班链条上插入本航班
                df = pd.DataFrame(self.env)
                # 获取先导航班(起飞时间小于本航班起飞时间，起飞时间从大到小排列的第一条记录
                r_ = df[(df[10] == new_planeID) & (df[6] < time_d)].sort_values(by=6, ascending=False)
                # 如果有先导航班
                if len(r_) > 0:
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
                    # 违反过站时间硬约束，如果之前已经违反了那么不加1，否则fault+1
                    if pass_time < 50:
                        if new_row_pre[58] == 0:
                            self.fault += 1
                            self.loss_val[0] = self.fault
                            new_row_pre[58] = 1
                    # 更新过站时间
                    new_row_pre[45] = pass_time

                    # 是否存在后继航班
                    if new_id_next > 0:
                        new_row_next = self.env[new_id_next - 1]
                        # 更新本航班的后继航班
                        row[44] = new_id_next
                        # 后继航班的先导航班就是本航班
                        new_row_next[43] = lineID
                        # 更新后继航班的过站时间
                        pass_time = new_row_next[6] - row[8]
                        # 违反过站时间硬约束，如果之前已经违反了那么不加1，否则fault+1
                        if pass_time < 50:
                            if row[58] == 0:
                                self.fault += 1
                                self.loss_val[0] = self.fault
                                row[58] = 1
                        # 更新过站时间
                        row[45] = pass_time
                    else:
                        # 不存在后继航班，那么本航班就成为了边际航班-最晚(也就是说原先那个先导航班就是原边际航班，现在被取代之)
                        # 这时需要检查边际约束-最晚，到达机场必须和原先本飞机最晚到达机场一致
                        # 原先的降落机场
                        airport = new_row_pre[5]
                        # 更新后继航班=0
                        row[44] = 0
                        # 更新是否是边际航班-最晚
                        row[66] = 1
                        # 原先先导航班不再是边际航班了
                        new_row_pre[66] = 0
                        # 降落机场不一致，违反硬约束，但是后续可能被修正，所以作记录但是不退出
                        if row[5] != airport:
                            # 原先已经是违反边际约束的不加1 否则fault+1
                            if row[64] == 0:
                                self.fault += 1
                                self.loss_val[0] = self.fault
                                row[64] = 1
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
                        # 更新后继航班的过站时间
                        pass_time = new_row_next[6] - row[8]
                        # 违反过站时间硬约束，如果之前已经违反了那么不加1，否则fault+1
                        if pass_time < 50:
                            if row[58] == 0:
                                self.fault += 1
                                self.loss_val[0] = self.fault
                                row[58] = 1
                        # 更新过站时间
                        row[45] = pass_time

                        # 检查边际约束，起飞机场必须与原先起飞机场一致
                        # 原先起飞机场
                        airport = new_row_next[4]
                        # 更新先导航班=0
                        row[43] = 0
                        # 更新是否是边际航班-最早
                        row[65] = 1
                        # 后继航班不再是边际航班了-最早
                        new_row_next[65] = 0
                        # 起飞机场不一致，违反硬约束，但是后续可能被修正，所以作记录但是不退出
                        if row[4] != airport:
                            # 原先已经是违反边际约束的不加1 否则fault+1
                            if row[63] == 0:
                                self.fault += 1
                                self.loss_val[0] = self.fault
                                row[63] = 1

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


    # 动作处理：调机
    def do_action_emptyflights(self, lineID):
        return lineID


    def show(self):
        print('loss_val: ', self.loss_val)
        print('fault: ', self.fault)
        print('action_log: ', self.action_log)
        print('action_count: ', self.action_count)

