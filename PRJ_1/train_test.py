# coding:utf-8
import numpy as np
import pandas as pd
import datetime
import os
import random
#import tensorflow as tf
#import tensorflow.contrib as tfc
import matplotlib.pyplot as plt

arr1 = np.array([
    [11],
    [12],
    [13],
    [14],
    [15],
    [16]
    ])

arr2 = np.array([
    [21],
    [22],
    [23],
    [24],
    [25],
    [26]
    ])

x_ = np.array([arr1, arr2])

print(x_.reshape(-1))

