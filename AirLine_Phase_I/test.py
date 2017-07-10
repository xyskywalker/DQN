import numpy as np
from AirLine_Phase_I.DataReader import DataReader
from AirLine_Phase_I.Environment import Environment
import datetime

# loss的参数，需要调整的即失效航班使用一个非常大的参数，原目标函数的参数一律除以100处理，用以加大与失效航班的差异
# 0, 100000:失效/故障/台风
# 1, 50:调机
# 2, 10:取消
# 3, 10:机型发生变化
# 4, 7.5:联程拉直
# 5, 1:延误
# 6, 1.5:提前
loss_para = np.array([100000, 50, 10, 10, 7.5, 1, 1.5] ,dtype=np.float32)

reader = DataReader(filename='DATA_20170705.xlsx' , env_d=65)
envObj = Environment(reader.arr_env, 1000, 100, 5,
                     reader.df_fault, reader.df_limit, reader.df_close, reader.df_flytime, reader.base_date)

envObj.show()

print(datetime.datetime.now(), 'Start - Action')
action = np.array([1, 1, 0, 0, 0, 0, 0])
envObj.step(action=action)
print(datetime.datetime.now(), 'End - Action')

envObj.show()

print(datetime.datetime.now(), 'Start - Action')
action = np.array([1, 1, 0, 0, 0, 0, 0])
envObj.step(action=action)
print(datetime.datetime.now(), 'End - Action')

envObj.show()

envObj.reset()
envObj.show()

# print(datetime.datetime.now(), 'Start - Read')
# print(reader.read())
# print(datetime.datetime.now(), 'End - Read')

default = 5

arr1 = np.array([1, 1, 0, 0])
arr2 = np.array([0, 1, 0, 1])
arr3 = arr1.copy()
arr3 += 1
print(arr3)
print(arr1)


