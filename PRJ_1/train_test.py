# coding:utf-8
import numpy as np
import pandas as pd
import datetime
import os

time_start = datetime.datetime(year=2017, month=7, day=29)
time_end = datetime.datetime(year=2016, month=5, day=31)

print(time_start.day)
print(time_start.weekday())
i = 2 if ((2 == 1) | (2 == 2)) else 1
print(i)
#time_start += datetime.timedelta(days=300)
#if time_start >= time_end:
#    print('>')
