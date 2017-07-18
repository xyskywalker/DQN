# coding:utf-8

import numpy as np
import tensorflow as tf
import tensorflow.contrib as tfc
import gym

env = gym.make('CartPole-v0')
env.reset()
'''
random_episodes = 0
reward_sum = 0
while random_episodes < 10:
    env.render()
    observation, reward, done, _ = env.step(np.random.randint(0,2))
    reward_sum += reward
    if done:
        random_episodes += 1
        print('Reward for this episode was: %g' % reward_sum)
        reward_sum = 0
        env.reset()
'''

# 网络参数
# 隐含层节点数
H = 50
batch_size = 25
learning_rate = 1e-1
# 环境信息维度
D = 4
# Reward的discount比例
gamma = 0.99

# discount rewards函数
def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

# 定义网络
# 输入
observations = tf.placeholder(tf.float32, [None, D], name='input_x')
W1 = tf.get_variable('W1', shape=[D, H], initializer=tfc.layers.xavier_initializer())
# 隐含层，没有偏置
layer1 = tf.nn.relu(tf.matmul(observations, W1))

W2 = tf.get_variable('W2', shape=[H, 1], initializer=tfc.layers.xavier_initializer())
score = tf.matmul(layer1, W2)
# 输出，往左还是往右
probability = tf.nn.sigmoid(score)

# 优化器使用Adam算法
adam = tf.train.AdamOptimizer(learning_rate=learning_rate)

# 虚拟的label
input_y = tf.placeholder(tf.float32, [None, 1], name='input_y')
# 每个Action的潜在价值
advantages = tf.placeholder(tf.float32, name='reward_signal')

# Action取值为1的概率为probability(即策略网络的输出概率)，Action取0的概率为1-probability
# label取值设定与Action相反，即label = 1-Action
# loglik就是当前Action对应的概率的对数
loglik = tf.log(input_y*(input_y - probability) + (1 - input_y)*(input_y + probability))
# 将loglik与潜在价值advantages相乘再取负数作为损失
loss = -tf.reduce_mean(loglik * advantages)

# 所有可训练的参数
tvars = tf.trainable_variables()
newGrads = tf.gradients(loss, tvars)

# 两层网络参数的梯度placeholder
W1Grad = tf.placeholder(tf.float32, name='batch_grad1')
W2Grad = tf.placeholder(tf.float32, name='batch_grad2')
batchGrad = [W1Grad, W2Grad]
updateGrads = adam.apply_gradients(zip(batchGrad, tvars))

# xs: 环境信息observation列表,即输入
# ys: label列表
# drs: 每一个action的reward
xs, ys, drs = [], [], []
reward_sum = 0
episode_number = 1
total_episodes = 10000

with tf.Session() as sess:
    rendering = False
    init = tf.global_variables_initializer()
    sess.run(init)
    observation = env.reset()

    gradBuffer = sess.run(tvars)
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    while episode_number <= total_episodes:
        if reward_sum/batch_size > 190 or rendering == True:
            #env.render()
            rendering = True
        x = np.reshape(observation, [1, D])

        tfprob = sess.run(probability, feed_dict={observations: x})
        action = 1 if np.random.uniform() < tfprob else 0

        xs.append(x)
        y = 1- action
        ys.append(y)

        observation, reward, done, info = env.step(action)
        reward_sum += reward
        drs.append(reward)

        if done:
            episode_number += 1
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            xs, ys, drs = [], [], []

            # 计算每一步Action的潜在价值，然后减去均值再除以标准差，得到一个零均值标准差为1的分布，有利于训练的稳定
            discounted_epr = discount_rewards(epr)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            # print('input_y', epy)
            # print('advantages', discounted_epr)
            tGrad = sess.run(newGrads, feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})
            for ix, grad in enumerate(tGrad):
                gradBuffer[ix] += grad

            if episode_number % batch_size == 0:
                sess.run(updateGrads, feed_dict={W1Grad: gradBuffer[0], W2Grad: gradBuffer[1]})
                for ix, grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0

                print('Average reward for episode %d : %f.' % (episode_number, reward_sum/batch_size))

                if reward_sum/batch_size >= 200:
                    print('Task solved in %d episodes!' % episode_number)
                    break

                reward_sum = 0

            observation = env.reset()

