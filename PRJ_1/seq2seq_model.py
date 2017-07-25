import tensorflow as tf  # Version 1.0 or 0.12
import tensorflow.contrib as tfc
import numpy as np
import matplotlib.pyplot as plt
import random
import math

train_data = np.load('train_data_1.npy')

def normalize(X, Y=None):
    """
    Normalise X and Y according to the mean and standard deviation of the X values only.
    """
    # # It would be possible to normalize with last rather than mean, such as:
    # lasts = np.expand_dims(X[:, -1, :], axis=1)
    # assert (lasts[:, :] == X[:, -1, :]).all(), "{}, {}, {}. {}".format(lasts[:, :].shape, X[:, -1, :].shape, lasts[:, :], X[:, -1, :])
    meanX = np.expand_dims(np.average(X, axis=1) + 0.00001, axis=1)
    stddevX = np.expand_dims(np.std(X, axis=1) + 0.00001, axis=1)
    #print (meanX.shape, stddevX.shape)
    #print (X.shape, Y.shape)

    meanY = np.expand_dims(meanX[:,:,6], axis=1)
    stddevY = np.expand_dims(stddevX[:,:,6], axis=1)


    X = X - meanX
    X = X / (2.5 * stddevX)
    if Y is not None:
        # print (meanX.shape, stddevX.shape)
        # print (X.shape, Y.shape)
        Y = Y - meanY
        Y = Y / (2.5 * stddevY)
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

        sig1 = train_data[rand: rand + seq_length * 2, 1:9]

        x1 = sig1[:seq_length, 0]
        x2 = sig1[:seq_length, 1]
        x3 = sig1[:seq_length, 2]
        x4 = sig1[:seq_length, 3]
        x5 = sig1[:seq_length, 4]
        x6 = sig1[:seq_length, 5]
        x8 = sig1[:seq_length, 7]
        y1 = sig1[seq_length:, 7]

        #x_ = np.array([x1])
        x_ = np.array([x1, x2, x3, x4, x5, x6, x8])
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
    batch_x, batch_y = normalize(batch_x, batch_y)
    return batch_x, batch_y

sample_x, sample_y = generate_x_y_data(isTrain=True, batch_size=3)
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
hidden_dim = 100  # Count of hidden neurons in the recurrent units.
# Number of stacked recurrent cells, on the neural depth axis.
layers_stacked_count = 5

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


def train_batch(batch_size):
    """
    Training step that optimizes the weights
    provided some batch_size X and Y examples from the dataset.
    """
    X, Y = generate_x_y_data(isTrain=True, batch_size=batch_size)
    feed_dict = {enc_inp[t]: X[t] for t in range(len(enc_inp))}
    feed_dict.update({expected_sparse_output[t]: Y[
                     t] for t in range(len(expected_sparse_output))})
    _, loss_t = sess.run([train_op, loss], feed_dict)
    return loss_t


def test_batch(batch_size):
    """
    Test step, does NOT optimizes. Weights are frozen by not
    doing sess.run on the train_op.
    """
    X, Y = generate_x_y_data(isTrain=False, batch_size=batch_size)
    feed_dict = {enc_inp[t]: X[t] for t in range(len(enc_inp))}
    feed_dict.update({expected_sparse_output[t]: Y[
                     t] for t in range(len(expected_sparse_output))})
    loss_t = sess.run([loss], feed_dict)
    return loss_t[0]


# Training
train_losses = []
test_losses = []

sess.run(tf.global_variables_initializer())

train_loss_all = 0.0
for t in range(nb_iters + 1):
    train_loss = train_batch(batch_size)
    train_losses.append(train_loss)
    train_loss_all += train_loss

    if t % 10 == 0:
        # Tester
        test_loss = test_batch(batch_size)
        test_losses.append(test_loss)
        print("Step {}/{}, train loss: {}, \tTEST loss: {}".format(t,
                                                                   nb_iters, train_loss_all / 10.0, test_loss))
        train_loss_all = 0.0

print("Fin. train loss: {}, \tTEST loss: {}".format(train_loss, test_loss))


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

X, Y = generate_x_y_data(isTrain=False, batch_size=nb_predictions)
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
