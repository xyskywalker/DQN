# coding:utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import requests
import io

f = open('航班表.csv')
df = pd.read_csv(f)
df = df.fillna(value=1.0)

print(df)