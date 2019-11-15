import tensorflow as tf
import random
import ipdb
import numpy as np

# ==========
#   MODEL
# ==========
class MLP:
    def __init__(self, input_dim, output_dim, learning_rate=0.1, mem_size=32, vocab_size=-1, model_id=None, saved_folder=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.mem_size = mem_size
        self.vocab_size = vocab_size

        #global count
        self.global_step = tf.train.get_or_create_global_step()
        self.increment_global_step = self.global_step.assign_add(1)
        self.model_id = model_id

        self.summary_dir = summary_dir if summary_dir[-1] == '/' else summary_dir + '/'
        self.summary_writer = tf.summary.FileWriter(self.summary_dir + self.model_id)


        return

    def setup(self):
        # ---------------------------------------------------------------------------------------------------------------
        # Input:
        self.x = tf.placeholder(shape=[None, self.input_dim], dtype=tf.int32)
        x_onehot = tf.one_hot(self.x, self.vocab_size, dtype=tf.float32)
        input_size = tf.shape(self.x)[1]*tf.shape(self.x)[2]
        x_input = tf.reshape(x_onehot, [-1, input_size])

        self.y = tf.placeholder(shape=[None], dtype=tf.int32)
        self.y_onehot = tf.one_hot(self.y, self.output_dim, dtype=tf.float32)

        # ---------------------------------------------------------------------------------------------------------------
        # Define weights for output decoding
        weights = {
            'hidden1': tf.Variable(tf.random_normal([input_size, self.mem_size])),
            'hidden2': tf.Variable(tf.random_normal([self.mem_size, self.mem_size])),
            'hidden3': tf.Variable(tf.random_normal([self.mem_size, self.output_dim]))
        }
        biases = {
            'hidden1': tf.Variable(tf.random_normal([self.mem_size])),
            'hidden2': tf.Variable(tf.random_normal([self.mem_size])),
            'hidden3': tf.Variable(tf.random_normal([self.output_dim]))
        }

        #---------------------------------------------------------------------------------------------------------------
        # Non-linear embeddings:
        hidden_layer    = tf.nn.elu(tf.matmul(tf.cast(self.x, tf.float32), weights['hidden1']) + biases['hidden1'])
        hidden_layer    = tf.nn.elu(tf.matmul(hidden_layer, weights['hidden2']) + biases['hidden2'])
        self.logits     = tf.matmul(hidden_layer, weights['hidden3']) + biases['hidden3']
        self.probs      = tf.nn.softmax(self.logits, axis=1)
        self.y_pred      = tf.argmax(self.probs, axis=1)

        # ---------------------------------------------------------------------------------------------------------------
        # Optimization:
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_onehot,
                                                                           logits=self.logits))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        #---------------------------------------------------------------------------------------------------------------
        # Metrics:
        correct_preds   = tf.equal(tf.argmax(self.probs, axis=1),
                                   tf.argmax(self.y_onehot, axis=1))
        self.precision  = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
        return

    # def step_validation(self.sess, x, y, clear_memory=True):
    #
    # def model.clear_memory(self.sess):
    #
    # def model.step_inference():

    def step_training(self, sess, x, y):
        episode_step = sess.run(self.global_step)
        ops = [self.loss, self.optimizer, self.y_pred]
        
        losses, y_preds = [], []
        for xx, yy in zip(x, y):
            outputs = sess.run(ops, feed_dict={self.x: xx[:, :self.input_dim],
                                               self.y: yy})
            loss_train = outputs[0]
            y_pred = outputs[2]

            losses.append(loss_train)
            y_preds.append(y_pred)

            sess.run(self.increment_global_step)
            episode_step += 1
        return losses, y_preds

    def step_inference(self, sess, x):
        ops = [self.y_pred]
        y_preds = []
        for xx in x:
            outputs = sess.run(ops, feed_dict={self.x: xx})
            y_pred = outputs[0]
            y_preds.append(y_pred)
        return y_preds

    def step_validation(self, sess, x, y):
        ops = [self.y_pred]
        y_preds = []
        for xx, yy in zip(x, y):
            outputs = sess.run(ops, feed_dict={self.x: xx[:, :self.input_dim]})
            y_pred = outputs[0]
            y_preds.append(y_pred)
        return y_preds



