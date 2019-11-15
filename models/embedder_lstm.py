'''
Implementation of LSTM with self-attention
@author: Omar U. Florez
'''

from __future__ import print_function

import tensorflow as tf
import random
import ipdb
import numpy as np

# ==========
#   MODEL
# ==========
class dynamicRNN:
    def __init__(self, seq_max_len, state_size, vocab_size, matrix_format=True, run_attention=False):
        self.seq_max_len = seq_max_len
        self.state_size = state_size
        self.vocab_size = vocab_size
        # self.num_classes = num_classes
        # when matrix_format is True, the input format is: (batch_size, seq_len)
        #   else, the input format is like a tensor: (batch_size, seq_len, vocab_size)
        self.matrix_format = matrix_format
        self.run_attention = run_attention
        return

    def build_model(self, x):
        # ---------------------------------------------------------------------------------------------------------------
        # Input:
        if self.matrix_format:
            self.x = x
            # [batch_size, seq_len] -> [batch_size, seq_len, vocab_size] (256, 2086)
            x_one_hot = tf.one_hot(tf.to_int32(self.x), self.vocab_size)
        else:
            # (batch_size, seq_len, voc_size)    (256, 32, 65)
            self.x = x
            x_one_hot = self.x

        # [batch_size, seq_len, vocab_size] -> seq_len x [batch_size, vocab_size]
        x_one_hot = tf.cast(x_one_hot, tf.float32)
        rnn_input = tf.unstack(x_one_hot, axis=1)

        # self.y = tf.placeholder(shape=[None], dtype=tf.int32)
        # self.y_onehot = tf.one_hot(self.y, self.num_classes, dtype=tf.float32)

        # A placeholder for indicating each sequence length
        self.seqlen = tf.placeholder(tf.int32, [None])
        ##self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')

        # ---------------------------------------------------------------------------------------------------------------
        # Define weights for output decoding
        weights = {
            'out': tf.Variable(tf.random_normal([self.state_size, self.state_size])),
            #'out2': tf.Variable(tf.random_normal([256, self.num_classes]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([self.state_size])),
            #'out2': tf.Variable(tf.random_normal([self.num_classes]))
        }

        # ---------------------------------------------------------------------------------------------------------------
        # LSTM params
        ##init_state = tf.zeros([self.batch_size, self.state_size])
        cell = tf.contrib.rnn.BasicLSTMCell(self.state_size)
        # Get lstm cell output, providing 'sequence_length' will perform dynamic calculation.
        # outputs: seq_len x [batch_size, state_size]
        # Returns:
        #   - outputs is a length T list of outputs (one for each input), a nested tuple of such elements.
        #   - state is the final state
        #   - cell.state_size: LSTMStateTuple(c=256, h=256)

        # initial_state = cell.zero_state(self.batch_size, tf.float32)
        # outputs:       seq_len x (batch_size, state_size)
        # finall_state:  c=(?, 256), h=(?, 256)
        self.outputs, self.final_state = tf.contrib.rnn.static_rnn(cell, rnn_input,
                                                                   dtype=tf.float32
                                                                   # initial_state=init_state
                                                                   # sequence_length=self.seqlen
                                                                   )

        # ---------------------------------------------------------------------------------------------------------------
        # When performing dynamic calculation, we must retrieve the last
        # dynamically computed output, i.e., if a sequence length is 10, we need
        # to retrieve the 10th output.
        # However TensorFlow doesn't support advanced indexing yet, so we build
        # a custom op that for each sample in batch size, get its length and
        # get the corresponding relevant output.

        # ---------------------------------------------------------------------------------------------------------------
        # # 'outputs' is a list of output at every timestep, we pack them in a Tensor
        # # and change back dimension to [batch_size, n_step, n_input]
        # outputs = tf.stack(outputs)
        # outputs = tf.transpose(outputs, [1, 0, 2])
        #
        # # Hack to build the indexing and retrieve the right output.
        # batch_size = tf.shape(outputs)[0]
        # # Start indices for each sample
        # index = tf.range(0, batch_size) * self.seq_max_len + (self.seqlen - 1)
        # # Indexing
        # # pre-outputs: (batch, 20, 64)
        # outputs = tf.gather(tf.reshape(outputs, [-1, self.state_size]), index)
        # post-outputs: (batch, 64)

        # WE CAN ALSO GET THE LAST OUTPUT SINCE IT CARRIES TEMPORAL INFORMATIONs
        # outputs = states[1]
        # [batch_size, state_size]
        if not self.run_attention:
            output = self.outputs[-1]
        else:
            output, self.alpha, self.eij = self.attention()

        print('self.run_attention: ', self.run_attention)

        # ---------------------------------------------------------------------------------------------------------------
        # Linear activation, using outputs computed above
        #self.hidden_layer = tf.matmul(output, weights['out']) + biases['out']
        self.hidden_layer = output
        #self.pred = tf.matmul(self.hidden_layer, weights['out2']) + biases['out2']

        # ---------------------------------------------------------------------------------------------------------------
        # Evaluate model
        # correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y_onehot, 1))
        # self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        #return self.pred
        return self.hidden_layer

    # def step_training(self, learning_rate=0.01):
    #     # Define loss and optimizer
    #     cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y_onehot))
    #     optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    #     return cost, optimizer


    def attention(self):
        self.current_weight = tf.Variable(tf.truncated_normal([self.state_size], -0.1, 0.1))
        self.past_weight = tf.Variable(tf.truncated_normal([self.state_size], -0.1, 0.1))
        self.output_weight = tf.Variable(tf.truncated_normal([self.state_size+self.state_size, self.state_size], -0.1, 0.1))

        scores = []
        current_embed = tf.multiply(self.outputs[-1], self.current_weight)

        for i in range(self.seq_max_len - 1):
            scores.append(tf.multiply(self.outputs[i], self.past_weight) + current_embed)
        #scores:    [batch_size, seq_len-1, state_size]
        scores = tf.reshape(tf.concat(scores, axis=1), [-1, self.seq_max_len-1, self.state_size])
        #eij:       [batch_size, seq_len-1, state_size]
        score_mean, score_var = tf.nn.moments(scores, [0])
        scores_hat = (scores - score_mean)/tf.sqrt(score_var + 0.00001)
        eij = tf.tanh(scores_hat)

        #Softmax across the unrolling dimension
        #alpha:    [batch_size, seq_len-1, state_size]
        alpha = tf.nn.softmax(eij, dim=1)
        past_outs = tf.reshape(tf.concat(self.outputs, axis=1), [-1, self.seq_max_len, self.state_size])
        past_outs = tf.slice(past_outs, [0,0,0], [-1, self.seq_max_len-1, self.state_size])
        #remove temporal axis
        #context:   [batch_size, seq_len, state_size] -> [batch_size, state_size]
        c = tf.reduce_sum(tf.multiply(alpha, past_outs), axis=1)
        new_output = tf.tanh(tf.matmul(tf.concat([c, self.outputs[-1]], axis=1), self.output_weight))
        return new_output, alpha, eij

