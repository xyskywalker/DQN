# coding:utf-8

import numpy as np
import itertools
import random
import scipy.misc
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib as tfc
import os


# 环境内物体对象的class
class gameOb():
    # coordinates:坐标
    # size:尺寸
    # intensity:亮度
    # channel:RGB颜色通道
    # reward:奖励值
    # name:名称
    def __init__(self, coordinates, size, intensity, channel, reward, name):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.intensity = intensity
        self.channel = channel
        self.reward = reward
        self.name = name


# 环境class
class gameEnv():
    def __init__(self, size):
        self.sizeX = size
        self.sizeY = size
        self.actions = 4
        self.object = []
        a = self.reset()
        plt.imshow(a, interpolation='nearest')
        # plt.show()


    # 重置
    def reset(self):
        self.objects = []
        hero = gameOb(self.newPosition(), 1, 1, 2, None, 'hero')
        self.objects.append(hero)
        goal = gameOb(self.newPosition(), 1, 1, 1, 1, 'goal')
        self.objects.append(goal)
        hole = gameOb(self.newPosition(), 1, 1, 0, -1, 'fire')
        self.objects.append(hole)
        goal2 = gameOb(self.newPosition(), 1, 1, 1, 1, 'goal')
        self.objects.append(goal2)
        hole2 = gameOb(self.newPosition(), 1, 1, 0, -1, 'fire')
        self.objects.append(hole2)
        goal3 = gameOb(self.newPosition(), 1, 1, 1, 1, 'goal')
        self.objects.append(goal3)
        goal4 = gameOb(self.newPosition(), 1, 1, 1, 1, 'goal')
        self.objects.append(goal4)

        state = self.renderEnv()
        self.state = state
        return state


    # 移动英雄角色
    # direction:0,1,2,3 分别代表 上下左右
    def moveChar(self, direction):
        hero = self.objects[0]
        heroX = hero.x
        heroY = hero.y
        if direction == 0 and hero.y >= 1:
            hero.y -= 1
        if direction == 1 and hero.y <= self.sizeY-2:
            hero.y += 1
        if direction == 2 and hero.x >= 1:
            hero.x -= 1
        if direction == 3 and hero.x <= self.sizeX-2:
            hero.x += 1
        self.objects[0] = hero

    # 选择一个与现有物体不冲突的新位置
    def newPosition(self):
        iterables = [range(self.sizeX), range(self.sizeY)]
        points = []
        for t in itertools.product(*iterables):
            points.append(t)
        currentPositions = []
        for objectA in self.objects:
            if (objectA.x, objectA.y) not in currentPositions:
                currentPositions.append((objectA.x, objectA.y))
        for pos in currentPositions:
            points.remove(pos)

        location = np.random.choice(range(len(points)), replace=False)

        return points[location]

    # 检测hero是否碰触了goal或者fire
    def checkGoal(self):
        others = []
        for obj in self.objects:
            if obj.name == 'hero':
                hero = obj
            else:
                others.append(obj)

        for other in others:
            if hero.x == other.x and hero.y == other.y:
                self.objects.remove(other)
                if other.reward == 1:
                    self.objects.append(gameOb(self.newPosition(), 1, 1, 1, 1, 'goal'))
                    return other.reward, True
                else:
                    self.objects.append(gameOb(self.newPosition(), 1, 1, 0, -1, 'fire'))
                    return other.reward, False

        return 0.0, False

    # 生成环境
    def renderEnv(self):
        a = np.ones([self.sizeY+2, self.sizeX+2, 3])
        a[1:-1, 1:-1,:] = 0
        hero = None
        for item in self.objects:
            a[item.y + 1:item.y + item.size + 1, item.x + 1: item.x + item.size + 1, item.channel] = item.intensity
        b = scipy.misc.imresize(a[:, :, 0], [84, 84, 1], interp='nearest')
        c = scipy.misc.imresize(a[:, :, 1], [84, 84, 1], interp='nearest')
        d = scipy.misc.imresize(a[:, :, 2], [84, 84, 1], interp='nearest')
        a = np.stack([b, c, d], axis=2)
        return a

    # 执行一步Action
    def step(self, action):
        self.moveChar(action)
        reward, done = self.checkGoal()
        state = self.renderEnv()
        return state, reward, done


env = gameEnv(size=5)


# 定义DQN
class Qnetwork():
    def __init__(self, h_size):
        # 输入的是游戏环境，单个图像被扁平化: [n, 84*84*3] = [n, 21168] 需要恢复成[n, 84, 84, 3]
        self.scalarInput = tf.placeholder(shape=[None, 21168], dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 84, 84, 3])
        # 四层卷积
        self.conv1 = tfc.layers.convolution2d(inputs=self.imageIn,
                                              num_outputs=32,
                                              kernel_size=[8,8],
                                              stride=[4,4],
                                              padding='VALID',
                                              biases_initializer=None)
        self.conv2 = tfc.layers.convolution2d(inputs=self.conv1,
                                              num_outputs=64,
                                              kernel_size=[4,4],
                                              stride=[2,2],
                                              padding='VALID',
                                              biases_initializer=None)
        self.conv3 = tfc.layers.convolution2d(inputs=self.conv2,
                                              num_outputs=64,
                                              kernel_size=[3,3],
                                              stride=[1,1],
                                              padding='VALID',
                                              biases_initializer=None)
        self.conv4 = tfc.layers.convolution2d(inputs=self.conv3,
                                              num_outputs=512,
                                              kernel_size=[7,7],
                                              stride=[1,1],
                                              padding='VALID',
                                              biases_initializer=None)
        # 输出切分:Action带来的价值和环境本身的价值
        self.streamAC, self.streamVC = tf.split(self.conv4, 2, 3)
        # 扁平化
        self.streamA = tfc.layers.flatten(self.streamAC)
        self.streamV = tfc.layers.flatten(self.streamVC)
        # 线性全连接层权重
        self.AW = tf.Variable(tf.random_normal([h_size//2, env.actions]))
        self.VW = tf.Variable(tf.random_normal([h_size//2, 1]))

        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)

        self.Qout = self.Value + tf.subtract(self.Advantage,
                                             tf.reduce_mean(self.Advantage, reduction_indices=1, keep_dims=True)
                                             )
        self.predict = tf.argmax(self.Qout, 1)

        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.action_onehot = tf.one_hot(self.actions, env.actions, dtype=tf.float32)
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.action_onehot), reduction_indices=1)

        # 损失函数，均方差
        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        # 优化器
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)

