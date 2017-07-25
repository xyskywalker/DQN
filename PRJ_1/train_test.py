# coding:utf-8
import numpy as np
import pandas as pd
import datetime
import os
import tensorflow as tf
import tensorflow.contrib as tfc


train_data = np.load('train_data.npy')

print(train_data.shape)
print(train_data[0])
print(train_data[0].shape)

