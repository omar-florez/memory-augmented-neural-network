'''
CNN encoder
@author: Omar U. Florez
'''

import tensorflow as tf

class LeNet(object):
    '''
    Standard LeNet topology: 5 layers + 1 dense map
    '''

    def __init__(self, image_size, num_channels, hidden_dim):
        self.image_size = image_size
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        #to initialize matrix and biases variables
        self.matrix_init = tf.truncated_normal_initializer(stddev=0.1)
        self.vector_init = tf.constant_initializer(0.0)

    def core_builder(self, x):
        '''
        Embeded x using a CNN 
        :param x: image, a 2D tensor [batch_size, -1]
        :return: 2D tensor [batch_size, hidden_dim]
        '''
        number_feature_maps1 = 32*2
        number_feature_maps2 = 32*4

        #kernels
        conv1_weights = tf.get_variable('conv1_w',
                                        [3, 3, self.num_channels, number_feature_maps1],
                                        initializer=self.matrix_init)
        conv1_biases = tf.get_variable('conv1_b',
                                       [number_feature_maps1],
                                       initializer=self.vector_init)
        conv1a_weights = tf.get_variable('conv1a_w',
                                          [3, 3, number_feature_maps1, number_feature_maps1],
                                          initializer=self.matrix_init)
        conv1a_biases = tf.get_variable('conv1a_b',
                                         [number_feature_maps1],
                                         initializer=self.vector_init)
        conv2_weights = tf.get_variable('conv2_w',
                                        [3, 3, number_feature_maps1, number_feature_maps2],
                                        initializer=self.matrix_init)
        conv2_biases = tf.get_variable('conv2_b',
                                       [number_feature_maps2],
                                       initializer=self.vector_init)
        conv2a_weigths = tf.get_variable('conv2a_w',
                                         [3,3,number_feature_maps2, number_feature_maps2],
                                         initializer=self.matrix_init)
        conv2a_biases = tf.get_variable('conv2a_b',
                                        [number_feature_maps2],
                                        initializer=self.vector_init)
        fc1_weights = tf.get_variable('fc1_w',
                                      [self.image_size//4*self.image_size//4*number_feature_maps2, self.hidden_dim],
                                      initializer=self.matrix_init)
        fc1_biases = tf.get_variable('fc1_b',
                                     [self.hidden_dim],
                                     initializer=self.vector_init)

        #define models
        x = tf.reshape(x,[-1, self.image_size, self.image_size, self.num_channels])
        batch_size = tf.shape(x)[0]

        conv1 = tf.nn.conv2d(x, conv1_weights, strides=[1,1,1,1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
        conv1 = tf.nn.conv2d(conv1, conv1a_weights, strides=[1,1,1,1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1a_biases))

        pool1 = tf.nn.max_pool(relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1,1,1,1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
        conv2 = tf.nn.conv2d(relu2, conv2a_weigths, strides=[1,1,1,1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2a_biases))

        pool2 = tf.nn.max_pool(relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        reshape = tf.reshape(pool2, [batch_size, -1])
        hidden = tf.nn.bias_add(tf.matmul(reshape, fc1_weights), fc1_biases)

        return hidden







