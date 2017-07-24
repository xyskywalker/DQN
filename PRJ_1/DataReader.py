# coding:utf-8
import numpy as np
import pandas as pd
import datetime
import os

#lines = open('gy_contest_link_traveltime_training_data.txt').readlines()
#fp = open('training_data.txt','w')
#for s in lines:
#    # replace是替换，write是写入
#    fp.write( s.replace(';',',').replace('[','').replace(')','')
#              .replace('time_interval', 'time_interval-s,time_interval-e'))
#fp.close()  # 关闭文件

df_train = pd.read_csv('training_data.txt').sort_values(by=['link_ID', 'time_interval-s'])

#df_train[df_train['link_ID'] == 3377906287934510514].to_csv('3377906287934510514.csv')

#df_train[df_train['link_ID'] == 9377906282776510514].to_csv('9377906282776510514.csv')

df_link = pd.read_csv('gy_contest_link_info.txt', sep=';')
arr_link = np.array(df_link)
print('Data Loaded')

train_list = list(range(len(arr_link)))
link_id = 0

for row_link in arr_link:
    arr_train = np.array(df_train[df_train['link_ID'] == row_link[0]])
    time_start = datetime.datetime(year=2016, month=3, day=1)
    time_end = datetime.datetime(year=2016, month=5, day=31)
    row_index = 0
    days = 0
    all_index = 0
    # 3,4,5月，共92天，每天720段=66240，6月30天，每天60段=1800，共计66240+1800=68040
    # 列：0日期(相对于基准日期的第几天)，1日期(一月中的第几天)，2月份，3星期几，4是否周末，5小时，6分钟，
    # 7开始分钟数(总分钟数)，8通过时间，9路段长度，10路段宽度
    arr_temp = np.zeros([68040, 11], dtype=np.float32)
    while time_start <= time_end:
        # 一天应该有720段
        for i in range(720):
            time_min = (time_start + datetime.timedelta(minutes= i * 2))
            row = arr_train[row_index]
            if row[2] == time_min.strftime('%Y-%m-%d %H:%M:%S'):
                row_index += 1

            row_temp = arr_temp[all_index]
            row_temp[0] = float(days)
            row_temp[1] = float(time_start.day)
            row_temp[2] = float(time_start.month)
            row_temp[3] = float(time_start.weekday()) #周一=0，周日=6
            row_temp[4] = 1.0 if ((time_start.weekday() == 5) | (time_start.weekday() == 6)) else 0.0
            row_temp[5] = float(time_min.hour)
            row_temp[6] = float(time_min.minute)
            row_temp[7] = i * 2.0
            row_temp[8] = row[4]
            row_temp[9] = row_link[1]
            row_temp[10] = row_link[2]

            all_index += 1
        days += 1
        time_start += datetime.timedelta(days=1)

    time_end = datetime.datetime(year=2016, month=6, day=30)
    while time_start <= time_end:
        # 6月每天60段
        for i in range(60):
            time_min = (time_start + datetime.timedelta(minutes= i * 2 + 360))
            if row_index >= len(arr_train):
                row_index -= 1
            row = arr_train[row_index]
            if row[2] == time_min.strftime('%Y-%m-%d %H:%M:%S'):
                row_index += 1

            row_temp = arr_temp[all_index]
            row_temp[0] = float(days)
            row_temp[1] = float(time_start.day)
            row_temp[2] = float(time_start.month)
            row_temp[3] = float(time_start.weekday()) #周一=0，周日=6
            row_temp[4] = 1.0 if ((time_start.weekday() == 5) | (time_start.weekday() == 6)) else 0.0
            row_temp[5] = float(time_min.hour)
            row_temp[6] = float(time_min.minute)
            row_temp[7] = i * 2.0 + 360
            row_temp[8] = row[4]
            row_temp[9] = row_link[1]
            row_temp[10] = row_link[2]

            all_index += 1
        days += 1
        time_start += datetime.timedelta(days=1)

    print('link count', link_id, 'Link ID:', row_link[0])
    train_list[link_id] = arr_temp
    pd.DataFrame(arr_temp).to_csv('%20d.csv' % row_link[0])
    link_id += 1
np.save('train_data.npy', train_list)


