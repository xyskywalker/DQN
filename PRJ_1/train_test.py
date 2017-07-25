# coding:utf-8
import numpy as np
import pandas as pd
import datetime
import os
import random
#import tensorflow as tf
#import tensorflow.contrib as tfc
import matplotlib.pyplot as plt


#train_data = np.load('train_data.npy')

#print(train_data.shape)
#print(train_data[0])

#np.save('train_data_1.npy',train_data[0])

#train_data = np.load('train_data_1.npy')
#print(train_data.shape)
#print(train_data)


train_data = np.load('train_data_1.npy')
df_train_data = pd.DataFrame(train_data).sort_values(by=[7,0])

print(np.array(df_train_data)[0:30,8])
#rint(train_data[:,8])
plt.plot(np.array(df_train_data)[0:160,8])
plt.show()



