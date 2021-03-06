import tensorflow as tf  # Version 1.0 or 0.12
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

test_data_arr = []
test_data_mean = []
test_data_stddev = []
for train_data in train_data_all:
    df_train_data = pd.DataFrame(train_data).sort_values(by=[0, 7]).copy()
    df_test_data = pd.DataFrame(train_data).sort_values(by=[0, 7]).copy()

    df_train_data = df_train_data[df_train_data[2] < 6.0]
    df_test_data = df_test_data[df_test_data[2] >= 6.0]

    ##########Train Data##########
    train_data_ = np.array(df_train_data)
    train_data_[:,8] = train_data_[:,9] / train_data_[:,8] # 通过时间折算为速度
    mean = np.average(train_data_, axis=0) + 0.00001
    stddev = np.std(train_data_, axis=0) + 0.00001

    train_data_ = train_data_ - mean
    train_data_ = train_data_ / stddev

    train_data_arr.append(train_data_)
    train_data_mean.append(mean)
    train_data_stddev.append(stddev)

    ##########Test Data##########
    test_data_ = np.array(df_test_data)
    test_data_[:,8] = test_data_[:,9] / test_data_[:,8] # 通过时间折算为速度
    test_mean = np.average(test_data_, axis=0) + 0.00001
    test_stddev = np.std(test_data_, axis=0) + 0.00001

    test_data_ = test_data_ - test_mean
    test_data_ = test_data_ / test_stddev

    test_data_arr.append(test_data_)
    test_data_mean.append(test_mean)
    test_data_stddev.append(test_stddev)

print('train_data', len(train_data_arr))
print('train_data.shape', train_data_arr[0].shape)

def generate_x_y_data(isTrain, batch_size, linkIndex):
    seq_length = 30

    batch_x = []
    batch_y = []
    train_data = train_data_arr[linkIndex]
    test_data = test_data_arr[linkIndex]

    for b_ in range(batch_size):
        #处理哪天
        range_i = random.randint(0, 91)
        #每天中开始数
        i_start = range_i * 720
        rand = random.randint(i_start, i_start + 720 - seq_length * 2)
        sig1 = train_data[rand: rand + seq_length * 2,:]

        if isTrain is False:
            rand = 0
            sig1 = test_data[rand: rand + seq_length * 2, :]

        x1 = sig1[:seq_length, 1]
        x2 = sig1[:seq_length, 2]
        x3 = sig1[:seq_length, 3]
        x4 = sig1[:seq_length, 4]
        x5 = sig1[:seq_length, 5]
        x6 = sig1[:seq_length, 6]
        x7 = sig1[:seq_length, 7]
        x8 = sig1[:seq_length, 8] ########
        #x9 = sig1[:seq_length, 9]
        #x10 = sig1[:seq_length, 10]
        y1 = sig1[seq_length:, 8]

        #x_ = np.array([x1])
        x_ = np.array([x1, x2, x3, x4, x5, x6, x7, x8])#, x9, x10])
        y_ = np.array([y1])
        x_, y_ = x_.T, y_.T

        batch_x.append(x_)
        batch_y.append(y_)

    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
    # shape: (batch_size, seq_length, output_dim)

    batch_x = np.array(batch_x).transpose((1, 0, 2))
    batch_y = np.array(batch_y).transpose((1, 0, 2))
    # shape: (seq_length, batch_size, output_dim)
    return batch_x, batch_y

sample_x, sample_y = generate_x_y_data(isTrain=True, batch_size=3, linkIndex=0)
print("Dimensions of the dataset for 3 X and 3 Y training examples : ")
print(sample_x.shape)
print(sample_x)
print(sample_y.shape)
print(sample_y)
print("(seq_length, batch_size, output_dim)")

# Internal neural network parameters
# Time series will have the same past and future (to be predicted) lenght.
seq_length = sample_x.shape[0]
batch_size = 100  # Low value used for live demo purposes - 100 and 1000 would be possible too, crank that up!

# Output dimension (e.g.: multiple signals at once, tied in time)
input_dim = sample_x.shape[-1]
output_dim = sample_y.shape[-1]
hidden_dim = 35  # Count of hidden neurons in the recurrent units.
# Number of stacked recurrent cells, on the neural depth axis.
layers_stacked_count = 2

# Optmizer:
learning_rate = 0.007  # Small lr helps not to diverge during training.
# How many times we perform a training step (therefore how many times we
# show a batch).
nb_iters = 100000
lr_decay = 0.92  # default: 0.9 . Simulated annealing.
momentum = 0.5  # default: 0.0 . Momentum technique in weights update
lambda_l2_reg = 0.003  # L2 regularization of weights - avoids overfitting


# ## Definition of the seq2seq neuronal architecture
#
# <img src="https://www.tensorflow.org/images/basic_seq2seq.png" />
#
# Comparatively to what we see in the image, our neural network deals with
# signal rather than letters. Also, we don't have the feedback mechanism
# yet, which is to be implemented in the exercise 4. We do have the "GO"
# token however.

# In[4]:


# Backward compatibility for TensorFlow's version 0.12:
try:
    tf.nn.seq2seq = tf.contrib.legacy_seq2seq
    tf.nn.rnn_cell = tf.contrib.rnn
    tf.nn.rnn_cell.GRUCell = tf.contrib.rnn.GRUCell
    print("TensorFlow's version : 1.0 (or more)")
except:
    print("TensorFlow's version : 0.12")


# In[5]:


tf.reset_default_graph()
# sess.close()
sess = tf.InteractiveSession()

with tf.variable_scope('Seq2seq'):

    # Encoder: inputs
    enc_inp = [
        tf.placeholder(tf.float32, shape=(
            None, input_dim), name="inp_{}".format(t))
        for t in range(seq_length)
    ]

    # Decoder: expected outputs
    expected_sparse_output = [
        tf.placeholder(tf.float32, shape=(None, output_dim),
                       name="expected_sparse_output_".format(t))
        for t in range(seq_length)
    ]

    # Give a "GO" token to the decoder.
    # You might want to revise what is the appended value "+ enc_inp[:-1]".
    dec_inp = [tf.zeros_like(
        enc_inp[0], dtype=np.float32, name="GO")] + enc_inp[:-1]

    # Create a `layers_stacked_count` of stacked RNNs (GRU cells here).
    cells = []
    for i in range(layers_stacked_count):
        with tf.variable_scope('RNN_{}'.format(i)):
            cells.append(tf.nn.rnn_cell.GRUCell(hidden_dim))
            # cells.append(tf.nn.rnn_cell.BasicLSTMCell(...))
    cell = tf.nn.rnn_cell.MultiRNNCell(cells)

    # For reshaping the input and output dimensions of the seq2seq RNN:
    w_in = tf.Variable(tf.random_normal([input_dim, hidden_dim]))
    b_in = tf.Variable(tf.random_normal([hidden_dim], mean=1.0))
    w_out = tf.Variable(tf.random_normal([hidden_dim, output_dim]))
    b_out = tf.Variable(tf.random_normal([output_dim]))

    reshaped_inputs = [tf.nn.relu(tf.matmul(i, w_in) + b_in) for i in enc_inp]

    # Here, the encoder and the decoder uses the same cell, HOWEVER,
    # the weights aren't shared among the encoder and decoder, we have two
    # sets of weights created under the hood according to that function's def.
    dec_outputs, dec_memory =  tfc.legacy_seq2seq.basic_rnn_seq2seq(
        enc_inp,
        dec_inp,
        cell
    )

    output_scale_factor = tf.Variable(1.0, name="Output_ScaleFactor")
    # Final outputs: with linear rescaling similar to batch norm,
    # but without the "norm" part of batch normalization hehe.
    reshaped_outputs = [output_scale_factor *
                        (tf.matmul(i, w_out) + b_out) for i in dec_outputs]


# In[6]:


# Training loss and optimizer

with tf.variable_scope('Loss'):
    # L2 loss
    output_loss = 0
    for _y, _Y in zip(reshaped_outputs, expected_sparse_output):
        output_loss += tf.reduce_mean(tf.nn.l2_loss(_y - _Y))

    # L2 regularization (to avoid overfitting and to have a  better
    # generalization capacity)
    reg_loss = 0
    for tf_var in tf.trainable_variables():
        if not ("Bias" in tf_var.name or "Output_" in tf_var.name):
            reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

    loss = output_loss + lambda_l2_reg * reg_loss

with tf.variable_scope('Optimizer'):
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate, decay=lr_decay, momentum=momentum)
    train_op = optimizer.minimize(loss)


# ## Training of the neural net

# In[7]:


def train_batch(batch_size, linkIndex):
    """
    Training step that optimizes the weights
    provided some batch_size X and Y examples from the dataset.
    """
    X, Y = generate_x_y_data(isTrain=True, batch_size=batch_size, linkIndex=linkIndex)
    feed_dict = {enc_inp[t]: X[t] for t in range(len(enc_inp))}
    feed_dict.update({expected_sparse_output[t]: Y[
                     t] for t in range(len(expected_sparse_output))})
    _, loss_t = sess.run([train_op, loss], feed_dict)
    return loss_t


def test_batch(batch_size, linkIndex):
    """
    Test step, does NOT optimizes. Weights are frozen by not
    doing sess.run on the train_op.
    """
    X, Y = generate_x_y_data(isTrain=False, batch_size=batch_size, linkIndex=linkIndex)
    feed_dict = {enc_inp[t]: X[t] for t in range(len(enc_inp))}
    feed_dict.update({expected_sparse_output[t]: Y[
                     t] for t in range(len(expected_sparse_output))})
    loss_t = sess.run([loss], feed_dict)
    return loss_t[0]


def test_batch_mape(batch_size, linkIndex):
    X, Y = generate_x_y_data(isTrain=False, batch_size=batch_size, linkIndex=linkIndex)
    feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
    outputs = np.array(sess.run([reshaped_outputs], feed_dict)[0])

    stddev_test = test_data_stddev[linkIndex]
    mean_test = test_data_mean[linkIndex]

    outputs *= stddev_test[8]
    outputs += mean_test[8]

    Y *= stddev_test[8]
    Y += mean_test[8]

    mape_all = 0.0
    for j in range(batch_size):
        mape = 0.0
        for k in range(output_dim):
            expected = Y[:, j, k]
            pred = outputs[:, j, k]
            mape += sum(abs(pred - expected) / expected) / float(len(expected))
        mape_all += mape
    mape_all /= batch_size
    return mape_all


# Training
train_losses = []
test_losses = []

sess.run(tf.global_variables_initializer())

train_loss_all = 0.0
linkIndex = 130  #78,98,114,42
mape_all = 0.0
mape_count = 0
for t in range(nb_iters + 1):
    train_loss = train_batch(batch_size, linkIndex=linkIndex)
    train_losses.append(train_loss)
    train_loss_all += train_loss

    if t % 10 == 0:
        # Tester
        test_loss = test_batch_mape(batch_size, linkIndex=linkIndex)
        test_losses.append(test_loss)
        mape_all += test_loss
        mape_count += 1
        print("Step {}/{}, train loss: {}, \tTEST MAPE: {}, \t\tLink Index: {}".format(t,
                                                                   nb_iters, train_loss_all / 10.0, test_loss, linkIndex))
        train_loss_all = 0.0

    if mape_count == 10:
        print(' -=10 MAPEs ave=- ', mape_all / 10.0)
        mape_all = 0.0
        mape_count = 0

    linkIndex += 1
    if linkIndex >= 132:
        linkIndex = 0
# In[8]:


# Plot loss over time:
'''
plt.figure(figsize=(12, 6))
plt.plot(
    np.array(range(0, len(test_losses))) /
    float(len(test_losses) - 1) * (len(train_losses) - 1),
    np.log(test_losses),
    label="Test loss"
)
plt.plot(
    np.log(train_losses),
    label="Train loss"
)
plt.title("Training errors over time (on a logarithmic scale)")
plt.xlabel('Iteration')
plt.ylabel('log(Loss)')
plt.legend(loc='best')
plt.show()
'''

# In[9]:

# Test
nb_predictions = 5
print("Let's visualize {} predictions with our signals:".format(nb_predictions))

X, Y = generate_x_y_data(isTrain=False, batch_size=nb_predictions, linkIndex=15)
feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
outputs = np.array(sess.run([reshaped_outputs], feed_dict)[0])

for j in range(nb_predictions):
    plt.figure(figsize=(12, 3))

    for k in range(output_dim):
        past = X[:, j, 7]
        expected = Y[:, j, k]
        pred = outputs[:, j, k]

        label1 = "Seen (past) values" if k == 0 else "_nolegend_"
        label2 = "True future values" if k == 0 else "_nolegend_"
        label3 = "Predictions" if k == 0 else "_nolegend_"
        plt.plot(range(len(past)), past, "o--b", label=label1)
        plt.plot(range(len(past), len(expected) + len(past)),
                 expected, "x--b", label=label2)
        plt.plot(range(len(past), len(pred) + len(past)),
                 pred, "o--y", label=label3)

    plt.legend(loc='best')
    plt.title("Predictions v.s. true values")
    plt.show()

print("Reminder: the signal can contain many dimensions at once.")
print("In that case, signals have the same color.")
print("In reality, we could imagine multiple stock market symbols evolving,")
print("tied in time together and seen at once by the neural network.")


# ## Author
#
# Guillaume Chevalier
#
# - https://ca.linkedin.com/in/chevalierg
# - https://twitter.com/guillaume_che
# - https://github.com/guillaume-chevalier/
#

# In[10]:

# # Let's convert this notebook to a README for the GitHub project's title page:
# get_ipython().system('jupyter nbconvert --to markdown seq2seq.ipynb')
# get_ipython().system('mv seq2seq.md README.md')
