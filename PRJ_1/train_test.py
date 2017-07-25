# coding:utf-8
import numpy as np
import pandas as pd
import datetime
import os
import random
#import tensorflow as tf
#import tensorflow.contrib as tfc


#train_data = np.load('train_data.npy')

#print(train_data.shape)
#print(train_data[0])

#np.save('train_data_1.npy',train_data[0])

#train_data = np.load('train_data_1.npy')
#print(train_data.shape)
#print(train_data)


train_data = np.load('train_data_1.npy')

def normalize(X, Y=None):
    """
    Normalise X and Y according to the mean and standard deviation of the X values only.
    """
    # # It would be possible to normalize with last rather than mean, such as:
    # lasts = np.expand_dims(X[:, -1, :], axis=1)
    # assert (lasts[:, :] == X[:, -1, :]).all(), "{}, {}, {}. {}".format(lasts[:, :].shape, X[:, -1, :].shape, lasts[:, :], X[:, -1, :])
    mean = np.expand_dims(np.average(X, axis=1) + 0.00001, axis=1)
    stddev = np.expand_dims(np.std(X, axis=1) + 0.00001, axis=1)
    # print (mean.shape, stddev.shape)
    # print (X.shape, Y.shape)
    X = X - mean
    X = X / (2.5 * stddev)
    if Y is not None:
        assert Y.shape == X.shape, (Y.shape, X.shape)
        Y = Y - mean
        Y = Y / (2.5 * stddev)
        return X, Y
    return X

def generate_x_y_data(isTrain=True, batch_size=3):
    seq_length = 30

    batch_x = []
    batch_y = []
    for _ in range(batch_size):
        rand = random.randint(0, 66000 - seq_length * 2)

        if isTrain is False:
            rand = random.randint(66000, 66239 - seq_length * 2)

        sig1 = train_data[rand : rand + seq_length * 2, 1:8]

        x1 = sig1[:seq_length, 0]
        x2 = sig1[:seq_length, 1]
        x3 = sig1[:seq_length, 2]
        x4 = sig1[:seq_length, 3]
        x5 = sig1[:seq_length, 4]
        x6 = sig1[:seq_length, 5]
        x7 = sig1[:seq_length, 6]
        y1 = sig1[seq_length:,6]

        x_ = np.array([x1, x2, x3, x4, x5, x6, x7])
        y_ = np.array([y1])
        x_, y_ = x_.T, y_.T

        batch_x.append(x_)
        batch_y.append(y_)

    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
    # shape: (batch_size, seq_length, output_dim)

    print(batch_x.shape)
    print(batch_y.shape)
    batch_x = np.array(batch_x).transpose((1, 0, 2))
    batch_y = np.array(batch_y).transpose((1, 0, 2))
    print(batch_x.shape)
    # shape: (seq_length, batch_size, output_dim)
    # batch_x, batch_y = normalize(batch_x, batch_y)
    return batch_x, batch_y


generate_x_y_data(True, 5)



