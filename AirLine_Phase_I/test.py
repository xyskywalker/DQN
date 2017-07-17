import numpy as np
from AirLine_Phase_I.DataReader import DataReader
from AirLine_Phase_I.Environment import Environment
import datetime
import time
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

#reader.read(is_save=True, filename='env.npy')
env, fault, df_special_passtime = reader.read_fromfile(filename='env.npy')

envObj = Environment(reader.arr_env, 2364, 100, fault,
                     reader.df_fault, reader.df_limit, reader.df_close, reader.df_flytime, reader.base_date,
                     reader.df_plane_type, reader.df_first, reader.df_last, df_special_passtime,
                     reader.domestic_airport)

print(reader.time_d_max - reader.time_d_min)
time_d = 6000

print('fault', fault)
# dt = datetime.datetime.fromtimestamp(reader.base_date)

dt1 = reader.base_date.to_pydatetime()
dt2 = dt1 + datetime.timedelta(minutes=time_d)
#dt2 = datetime.datetime.fromtimestamp(time.time())

#print(dt2)
#print(dt2.hour)
#print(dt2.minute)
#rint(env[15])
#print(reader.arr_env[10])

#envObj.show()
'''
test_df = pd.DataFrame(env)
print(test_df[((test_df[13] ==1) | (test_df[16] == 1) | (test_df[25] == 1) | (test_df[29] == 1))
              & (test_df[46] == 1)
              & (test_df[2] == 1) & (test_df[5] == 50)])
'''

'''
end = False
i = 1
while end is False:
    print(datetime.datetime.now(), 'Start - Action')
    action = np.array([i, 4, 0, 0, 30, 0, 55])
    _, _ ,_, end = envObj.step(action=action)
    print(datetime.datetime.now(), 'End - Action')
    i += 1
'''

'''
print(datetime.datetime.now(), 'Start - Action')
for i in range(800):
    action = np.array([1, 1, 0, 0, 0, 0, 0])
    envObj.step(action=action)
    action = np.array([11, 2, 0, 0, 30, 0, 55])
    envObj.step(action=action)
    action = np.array([11, 3, 0, 0, 30, 0, 55])
    envObj.step(action=action)

print(datetime.datetime.now(), 'End - Action')
'''

print(datetime.datetime.now(), 'Start - Action')
action = np.array([0, 0, 48, 50, 1440, 40])
envObj.step(action=action)

print(datetime.datetime.now(), 'End - Action')
print('loss_val: ', envObj.loss_val)
print('fault: ', envObj.fault)
print('action_log: ', envObj.action_log)
print('action_log: ', envObj.action_log[0])
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

