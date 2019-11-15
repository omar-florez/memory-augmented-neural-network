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
    def __init__(self, data_args, encoder_args, matrix_format=True, run_attention=False):
        self.vocab_size_chars   = data_args['vocab_size_chars']
        self.vocab_size_words   = data_args['vocab_size_words']
        self.num_classes        = data_args['output_dim']

        self.state_size         = encoder_args['mem_size_encoder']
        self.bidirectional_rnn  = encoder_args['bidirectional_rnn']
        self.num_layers         = encoder_args['num_layers']
        self.encoder_size       = encoder_args['input_encoder_size']

        self.embedding_size     = self.state_size if not self.bidirectional_rnn else 2 * self.state_size

        # when matrix_format is True, the input format is: (batch_size, seq_len)
        #   else, the input format is like a tensor: (batch_size, seq_len, vocab_size)
        self.matrix_format      = matrix_format
        self.run_attention      = run_attention
        self.seqlen             = tf.placeholder(tf.int32, [None])

        self.weights = {
            'out': tf.Variable(tf.random_normal([self.embedding_size, self.state_size]), name='weights_0'),
            'out2': tf.Variable(tf.random_normal([self.state_size, self.num_classes]), name='weights_out'),
            #'output': tf.Variable(tf.random_normal([self.embedding_size, self.num_classes]), name='weights'),
        }
        self.biases = {
            'out': tf.Variable(tf.zeros([self.state_size]), name='bias_0'),
            'out2': tf.Variable(tf.zeros([self.num_classes]), name='bias_out')
            #'out': tf.Variable(tf.ones([self.state_size]), name='bias_0'),
            #'out2': tf.Variable(tf.ones([self.num_classes]), name='bias_out')
            # 'out': tf.Variable(tf.random_normal([self.state_size]), name='bias_0'),
            # 'out2': tf.Variable(tf.random_normal([self.num_classes]), name='bias_out'),
            #'output': tf.Variable(tf.random_normal([self.num_classes]), name='weights')
        }
 
    def setup(self, x, x_words, y, keep_prob=1.0, input_encoding='word_glove'):

        #----------------------------------------------------------------------------------------------------------------
        # label encoding
        self.y = y
        self.y_onehot = tf.one_hot(self.y, self.num_classes, dtype=tf.float32)

        #----------------------------------------------------------------------------------------------------------------
        # word-based encoding
        if input_encoding == 'word_glove':
            glove = np.load('./data/glove/glove_labels_dataset_zeros-noentries.npz')['embeddings']
            #glove_embedding         = np.vstack([glove, [[0.0]*300]])
            # glove_embedding = (80645, 300)
            glove_embedding         = tf.Variable(glove, dtype=tf.float32, trainable=False)

            # [None, seq_len] -> [None, seq_len, 300]
            word_embeddings     = tf.nn.embedding_lookup(glove_embedding, tf.cast(x_words, tf.int32))
            self.x_embeddings   = word_embeddings
        elif input_encoding == 'word_random':
            embedding           = tf.get_variable(name='embedding',
                                                 initializer=tf.truncated_normal([self.vocab_size_words, self.encoder_size]))
            word_embeddings     = tf.nn.embedding_lookup(embedding, tf.cast(x_words, tf.int32))
            self.x_embeddings   = word_embeddings
        #----------------------------------------------------------------------------------------------------------------
        # character-based encoding
        elif input_encoding == 'character_random':
            embedding           = tf.get_varible(name='embeding',
                                                 initializer=tf.truncated_normal([self.vocab_size_chars, self.encoder_size],
                                                                                 -0.0001, 0.0001))
            self.x_embeddings   = tf.nn.embedding_lookup(embedding, tf.cast(x, tf.int32))
        elif input_encoding == 'character_onehot':
            x_onehot = tf.one_hot(tf.cast(x, tf.int32), self.vocab_size_chars)
            self.x_embeddings   = x_onehot
            # x_embeddings = tf.transpose(x_embeddings, [1, 0, 2])
            # (?, seq_len, state_size) -> seq_len x (?, state_size)
            # rnn_input = tf.unstack(x_embeddings, axis=1)
            # self.seq_len = tf.reduce_sum(tf.sign(x), 1) * 0.0 + 32.0

        #----------------------------------------------------------------------------------------------------------------
        # Input dropout
        self.x_embeddings = tf.cast(self.x_embeddings, tf.float32)
        self.x_embeddings = tf.layers.dropout(self.x_embeddings, rate=keep_prob)

        #----------------------------------------------------------------------------------------------------------------
        # RNNs encoding
        with tf.variable_scope('RNN'):
            def make_cell():
                cell = tf.contrib.rnn.BasicLSTMCell(self.state_size)
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob) #,
                                                     #variational_recurrent=True, dtype=tf.float32)
                return cell

            # self.cell = tf.contrib.rnn.BasicLSTMCell(self.state_size)
            # self.outputs, self.final_state = tf.contrib.rnn.static_rnn(self.cell,
            #                                                            rnn_input,
            #                                                            dtype=tf.float32)
            if not self.bidirectional_rnn:
                self.cell = tf.contrib.rnn.MultiRNNCell([make_cell() for _ in range(self.num_layers)])
                # self.outputs: (?, seq_len, state_size)
                self.outputs, self.final_state = tf.nn.dynamic_rnn(self.cell,
                                                                   self.x_embeddings,
                                                                   dtype=tf.float32,
                                                                   time_major=False)
                                                                   #sequence_length=self.seq_len,
                                                                   #)
                # If time_major == False (default), this will be a Tensor shaped:
                #       [batch_size, max_time, cell.output_size]
                # output = self.outputs[-1]
                # self.outputs: (?, seq_len, state_size)
                output = self.outputs[:, -1, :]
                if self.run_attention:
                    output, self.alpha, self.eij = self.attention()
            else:
                fw_cell = tf.contrib.rnn.MultiRNNCell([make_cell() for _ in range(self.num_layers)])
                bw_cell = tf.contrib.rnn.MultiRNNCell([make_cell() for _ in range(self.num_layers)])

                #If time_major == False (default), this will be a Tensor shaped:
                #       [batch_size, max_time, cell.output_size]
                #self.outputs: 2x(?, seq_len, state_size)
                self.outputs, self.final_state = tf.nn.bidirectional_dynamic_rnn(fw_cell,
                                                                                 bw_cell,
                                                                                 self.x_embeddings,
                                                                                 dtype=tf.float32,
                                                                                 time_major=False)
                                                                               #sequence_length=self.seq_len,
                                                                               #)
                # sequence_length=self.seq_len,
                # )
                fw_output = self.outputs[0][:, -1, :]
                bw_output = self.outputs[1][:, 0, :]
                output = tf.concat([fw_output, bw_output], axis=1)

        self.hidden_layer = tf.nn.relu(tf.matmul(output, self.weights['out']) + self.biases['out'])
        self.logits = tf.matmul(self.hidden_layer, self.weights['out2']) + self.biases['out2']
        #logits = tf.matmul(output, self.weights['output']) + self.biases['output']
        self.probabilities = tf.nn.softmax(self.logits)

        y_pred = tf.argmax(self.probabilities, axis=1)
        #loss = self.compute_loss(self.logits, y)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=y))
        return loss, y_pred
       

    # def compute_loss(self, predictions, y, learning_rate=0.01):
    #     # Define loss and optimizer
    #     if y is None:
    #         return 0
    #     cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predictions, labels=y))
    #     #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    #     return cost

    def attention(self):
        #pdb.set_trace()
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
        eij = tf.tanh(scores)

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

