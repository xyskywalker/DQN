import numpy as np
from AirLine_Phase_I.DataReader import DataReader
from AirLine_Phase_I.Environment import Environment
import datetime
import pandas as pd


# loss的参数，需要调整的即失效航班使用一个非常大的参数，原目标函数的参数一律除以100处理，用以加大与失效航班的差异
# 0, 100000:失效/故障/台风
# 1, 50:调机
# 2, 10:取消
# 3, 10:机型发生变化
# 4, 7.5:联程拉直
# 5, 1:延误
# 6, 1.5:提前
loss_para = np.array([100000, 50, 10, 10, 7.5, 1, 1.5] ,dtype=np.float32)

reader = DataReader(filename='DATA_20170705.xlsx' , env_d=68)

env, fault, df_special_passtime = reader.read_fromfile(filename='env.npy')

envObj = Environment(reader.arr_env, 68, 1000, 100, fault,
                     reader.df_fault, reader.df_limit, reader.df_close, reader.df_flytime,
                     reader.base_date, reader.df_plane_type, reader.df_first, reader.df_last, df_special_passtime)

#print(reader.arr_env[10])

#envObj.show()

print(datetime.datetime.now(), 'Start - Action')
action = np.array([1, 1, 0, 0, 0, 0, 0])
envObj.step(action=action)
print(datetime.datetime.now(), 'End - Action')


print(datetime.datetime.now(), 'Start - Action')
action = np.array([11, 2, 0, 0, 3365, 0, 55])
envObj.step(action=action)
print(datetime.datetime.now(), 'End - Action')

print('loss_val: ', envObj.loss_val)
print('fault: ', envObj.fault)
print('action_log: ', envObj.action_log)
print('action_log: ', envObj.action_log[1])
print('action_count: ', envObj.action_count)
#envObj.reset()
#envObj.show()


#print(datetime.datetime.now(), 'Start - Read')
#print(reader.read(is_save = True, filename='env.npy'))
#print(datetime.datetime.now(), 'End - Read')

#arr1 = np.random.randint(low=1, high=10, size=[5, 3])
#df = pd.DataFrame(arr1)
#print(df[df[0] > 0].sort_values(by=2, ascending=False))

