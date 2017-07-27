# coding:utf-8
import numpy as np
import pandas as pd
import datetime
import os
import random
#import tensorflow as tf
#import tensorflow.contrib as tfc
import matplotlib.pyplot as plt


train_data = np.load('train_data.npy')

print(train_data.shape)
print(train_data[0])

np.save('train_data_1.npy',train_data[0])

train_data = np.load('train_data_1.npy')
print(train_data.shape)
print(train_data)

'''
train_data = np.load('train_data_1.npy')
df_train_data = pd.DataFrame(train_data).sort_values(by=[7,0])

plt.plot(np.array(df_train_data)[720 : 720 + 31, 8])
plt.plot(np.array(df_train_data)[720 * 1 : 720 * 1 + 31, 8])
plt.plot(np.array(df_train_data)[720 * 2 : 720 * 2 + 31, 8])
plt.plot(np.array(df_train_data)[720 * 3 : 720 * 3 + 31, 8])
plt.plot(np.array(df_train_data)[720 * 4 : 720 * 4 + 31, 8])
plt.show()

'''


