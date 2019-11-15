'''
Implementation of MLP
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
class MLP:
    def __init__(self, input_dim, key_dim):
        self.input_dim = input_dim
        self.key_dim = key_dim
        return

    def build_model(self, x):
        # ---------------------------------------------------------------------------------------------------------------
        # Input:
        self.x = x
        self.seqlen = tf.placeholder(tf.int32, [None])      #a horrible hack to make it complatible with the LSTM embedder

        # ---------------------------------------------------------------------------------------------------------------
        # Define weights for output decoding
        weights = {
            'hidden1': tf.Variable(tf.random_normal([self.input_dim, 128])),
            'hidden2': tf.Variable(tf.random_normal([128, self.key_dim]))
        }
        biases = {
            'hidden1': tf.Variable(tf.random_normal([128])),
            'hidden2': tf.Variable(tf.random_normal([self.key_dim]))
        }

        # ---------------------------------------------------------------------------------------------------------------
        # Linear activation, using outputs computed above
        self.hidden_layer_1 = tf.matmul(tf.cast(self.x, tf.float32), weights['hidden1']) + biases['hidden1']
        self.hidden_layer_2 = tf.matmul(self.hidden_layer_1, weights['hidden2']) + biases['hidden2']

        return self.hidden_layer_2

