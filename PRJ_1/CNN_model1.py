import tensorflow as tf
import tensorflow.contrib as tfc
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random
import math
import pandas as pd

train_data_all = np.load('train_data.npy')
train_data_arr = []
train_data_mean = []
train_data_stddev = []
for train_data in train_data_all:
    df_test_data = pd.DataFrame(train_data).sort_values(by=[7, 0])
    df_train_data = pd.DataFrame(train_data).sort_values(by=[7, 0])
    df_train_data = df_train_data[df_train_data[2] < 6.0]
    train_data_ = np.array(df_train_data)
    mean = np.average(train_data_, axis=0) + 0.00001
    stddev = np.std(train_data_, axis=0) + 0.00001

    train_data_ = train_data_ - mean
    train_data_ = train_data_ / stddev

    train_data_arr.append(train_data_)
    train_data_mean.append(mean)
    train_data_stddev.append(stddev)

print('train_data', len(train_data_arr))
print('train_data.shape', train_data_arr[0].shape)

train_data = train_data_arr[15]
line1 = train_data[92 * 6 : 92 * 6 + 92, 8]
plt.plot(line1)
line1 = train_data[92 * 7 : 92 * 7 + 92, 8]
plt.plot(line1)
line1 = train_data[92 * 8 : 92 * 8 + 92, 8]
plt.plot(line1)
line1 = train_data[92 * 9 : 92 * 9 + 92, 8]
plt.plot(line1)
plt.show()