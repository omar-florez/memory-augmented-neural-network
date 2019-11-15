'''
Model definition

Inspired from: Kaiser ???
'''

import tensorflow as tf
from models import embedder
from models import embedder_lstm
from models import embedder_mlp

from memories import memory
from memories import memory_continous

import ipdb
import numpy as np
from collections import Counter
from operator import itemgetter
import logging

from models import lstm_encoder


# Auxiliary methods:
logging.basicConfig(level=logging.INFO, filename='./myapp.log', filemode='w',)
def hist_str(array):
    output = sorted(Counter(np.array(array).ravel()).items(), key=itemgetter(0))
    return str(output)

class Model(object):

    def __init__(self,
                 data_args,
                 encoder_args,
                 memory_args,
                 learning_rate=1e-4,
                 model_id='lth_00_00.00',
                 summary_dir=None):

        # input data:
        self.seq_len_chars      = data_args['seq_len_chars']
        self.seq_len_words      = data_args['seq_len_words']
        self.output_dim         = data_args['output_dim']
        self.vocab_size_chars   = data_args['vocab_size_chars']
        self.vocab_size_words   = data_args['vocab_size_words']

        # encoder:
        self.num_layers         = encoder_args['num_layers']
        self.encoder_algo       = encoder_args['encoder_algo']
        self.input_encoding     = encoder_args['input_encoding']
        self.mem_size_encoder   = encoder_args['mem_size_encoder']
        self.encoder_size       = encoder_args['input_encoder_size']
        self.encoder            = self.get_encoder(data_args, encoder_args)

        # memories:
        self.mem_size           = memory_args['mem_size']
        self.key_dim            = memory_args['key_dim']
        self.memory_type        = memory_args['memory_type']
        self.keys_are_trainable = memory_args['keys_are_trainable']
        self.choose_k           = memory_args['choose_k']

        # get an external memories
        self.memory = self.get_memory()
        # self.classifier = self.get_classifier()

        # system:
        self.learning_rate = learning_rate
        self.global_step = tf.contrib.framework.get_or_create_global_step()
        self.increment_global_step = self.global_step.assign_add(1)
        self.model_id = model_id
        self.summary_dir = summary_dir if summary_dir[-1] == '/' else summary_dir + '/'
        self.summary_writer = tf.summary.FileWriter(self.summary_dir + self.model_id)

        # set up the resulting architecture which consists of a neural encoder
        # and a differentiable external memories
        self.setup()

    def get_encoder(self, data_args, encoder_args):
        if self.encoder_algo == 'cnn':
            return embedder.LeNet(self.image_size, number_channels, self.mem_size_encoder)
        elif self.encoder_algo == 'lstm':
            return lstm_encoder.dynamicRNN(data_args,
                                           encoder_args,
                                           run_attention=False)
        elif self.encoder_algo == 'mlp':
            return embedder_mlp.MLP(self.seq_len_chars, self.mem_size_encoder)
        return None

    def get_memory(self):
        '''
        Returns an external memories that will store hidden representations and that
        will respond to queries given a particular embedding vector while generating
        gradients to optimize the entire model.

        We consider two types of external memories:
            - Continous memories: GMM? (args.memory_type="lth_gmm")
            - Discrete memories: LTH? (args.memory_type="lth")
        '''
        if self.memory_type == 'lsh':
            mem = memory_lsh.Memory
        elif self.memory_type == 'lth_gmm':
           mem = memory_continous.Memory
        else:                    #lth
           mem = memory.Memory
        return mem(self.key_dim, self.mem_size, keys_are_trainable=self.keys_are_trainable, choose_k=self.choose_k)

    # def get_classifier(self):
    #     return Basic_Classifier(self.output_dim)

    def setup(self):
        '''
        Build the corresponding computational graph, which includes:
            - input: chars (self.x) and words (self.x_words)
            - neural encoders: MLP (models.mlp) and LSTM (models.model_lstm)
            - loss function:
            - external memories:
        '''

        #---------------------------------------------------------------------------------------------------------------
        # input placeholders:
        self.x          = tf.placeholder(tf.float32, [None, self.seq_len_chars], name='x')
        self.y          = tf.placeholder(tf.int32, [None], name='y')
        self.keep_prob  = tf.placeholder_with_default(1.0, shape=(), name='keep_prob')
        self.x_words    = tf.placeholder(tf.float32, [None, self.seq_len_words], name='x_words')

        #---------------------------------------------------------------------------------------------------------------
        # encoder:
        self.encoder.setup(self.x,
                           self.x_words,
                           self.y,
                           keep_prob=self.keep_prob,
                           input_encoding=self.input_encoding)

        self.embedding = self.encoder.hidden_layer
        self.embedding = tf.nn.dropout(self.embedding, keep_prob=self.keep_prob)

        #---------------------------------------------------------------------------------------------------------------
        # loss and gradients:
        self.y_pred_train, loss_hashing = self.memory.query(self.embedding, self.y, use_recent_idx=False)
        self.loss = loss_hashing
        self.gradient_ops = self.training_ops(self.loss)

        #---------------------------------------------------------------------------------------------------------------
        # external memories:
        (self.mem_keys, self.mem_vals, self.mem_age, self.recent_idx) = self.memory.get()
        self.mem_keys_reset = tf.placeholder(self.mem_keys.dtype, tf.identity(self.mem_keys).shape)
        self.mem_vals_reset = tf.placeholder(self.mem_vals.dtype, tf.identity(self.mem_vals).shape)
        self.mem_age_reset = tf.placeholder(self.mem_age.dtype, tf.identity(self.mem_age).shape)
        self.recent_idx_reset = tf.placeholder(self.recent_idx.dtype, tf.identity(self.recent_idx).shape)
        self.mem_reset_op = self.memory.set(self.mem_keys, self.mem_vals, self.mem_age, None)

    def training_ops(self, loss):
        opt = self.get_optimizer()
        params = tf.trainable_variables()

        #gradients = tf.gradients(self.memories.loss, params)
        #self.grad_summ = tf.summary.merge([tf.summary.histogram('%s-grad'%g.name, g)
        #                             for g in gradients if g is not None])

        print('Trainable params:')
        for param in params:
            print('\t', param)

        gradients = tf.gradients(self.loss, params)
        self.grad_summ = {"%s-%s"% (p.name, str(g.shape)): g for p, g in zip(params, gradients) if isinstance(g, tf.Tensor) }

        print('Available gradients:')
        for grads in self.grad_summ:
            print('\t', grads)

        # basic_lstm_cell/kernel:   [input_depth + h_depth, 4 * self._num_units]
        # basic_lstm_cell/bias:     [4 * self._num_units]
        # LSTM:
        #       h_t = W*h_t-1 + V*x_t
        #       stack two matrices for x and h_t-1
        #           x:      has size input_depth (vocab_size)
        #           h_t-1:  has size h_depth (_num_units)
        #       gate operations
        #           4 * self._num_units: 4 gates each one with self._num_units neurons
        #       get get gate outputs, split the big matrix into 4 parts
        #           i, j, f, o = array_ops.split(value=gate_inputs, num_or_size_splits=4, axis=one)
        ###gradients, _ = tf.clip_by_global_norm(gradients, 10.0)
        #This is the second part of `minimize()`. It returns an `Operation` that applies gradients
        return opt.apply_gradients(zip(gradients, params), global_step = self.global_step)

    def get_optimizer(self):
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-4)

    #-------------------------------------------------------------------------------------------------------------------
    #
    def step_training(self, sess, x, x_words, y, clear_memory=False, keep_prob=1.0):
        '''
        crunching numbers over batches: put training complexity here
        very batch is: (batch_size ,input_dim)
        :param sess: 
        :param x: A list of batches of observations defining the episode
        :param y: A list of batches of labels corresponding to x
        :param clear_memory: whether to clear the memories before an episode
        :return: 
        '''
        logging.info('-----------------------------------')
        logging.info('[TRAINING] label distribution in dataset:\t\t' + hist_str(y))

        memory_before = sess.run([self.mem_keys, self.mem_vals, self.mem_age])
        logging.info('[TRAINING] Memory value distribution (before):' + hist_str(memory_before[1]))
        if clear_memory:
            sess.run(self.memory.clear_memory())

        episode_count = sess.run(self.global_step)
        merged = tf.summary.merge_all()
        outputs = [
            self.loss, self.gradient_ops,
            self.y_pred_train,                           #2
            self.memory.mem_keys,                   #3
            self.memory.mem_vals,                   #4
            self.memory.mem_age,                    #5
            self.memory.recent_idx,                 #6
            self.memory.gradients,                  #7
            self.memory.positive_gradient,          #8
            self.memory.negative_gradient,          #9
            self.memory.loss,                       #10
            self.memory.incorrect_memory_lookup,    #11
            self.memory.nearest_neighbor_gradient,  #12
            self.memory.fetched_idxs,               #13
            self.memory.fetched_keys,               #14
            self.memory.fetched_vals,               #15

            self.memory.updated_fetched_keys,       #16
            self.memory.updated_idxs,               #17
            self.memory.updated_keys,               #18
            self.memory.updated_vals,               #19

            self.memory.oldest_idxs,                #20
            self.memory.normalized_x_val,           #21
            self.memory.y,                          #22

            self.memory.mem_age_incr,               #23
            self.memory.normalized_x,               #24

            self.memory.mem_query_proj,             #25
            self.embedding,                         #26

            self.memory.neighbors_ypred_vals,       #27
            self.memory.neighbors_ypred_sims,       #28
            self.memory.sims_temp,                  #29
            self.memory.neighbors_ypred_idxs,       #30
            self.memory.positive_gradient,          #31
            self.memory.negative_gradient,          #32
            self.memory.x,                          #33
            self.memory.x_pro,                      #34
            self.memory.normalized_x,               #35

            self.memory.num_hits,                   #36
            self.memory.num_misses,                 #37
            self.memory.positive_gradient_mem_idx,  #38
            #self.memories.pos_bias,
            #self.memories.neg_bias
            #self.memories.diff_gradient,
            self.grad_summ,                         #39
            #self.memories.positive_gradients,         #40
            #self.memories.largest_age                 #41
        ]

        losses = []
        y_preds = []
        summaries = []


        #len(y) : 100
        #len(xx): 10
        positive_gradients = []
        negative_gradients = []
        for xx, xx_words, yy in zip(x, x_words, y):
            outputs_ = sess.run(outputs, feed_dict={self.x:                 xx[:, :self.seq_len_chars],
                                                    self.x_words:           xx_words[:, :self.seq_len_words],
                                                    self.y:                 yy,
                                                    self.encoder.seqlen:   np.zeros(len(xx)),
                                                    self.keep_prob: keep_prob
                                                    })
            loss_train = outputs_[0]
            y_pred_train = outputs_[2]                  #(16,)
            mem_keys = outputs_[3]                      #(8192, 128)
            mem_vals = outputs_[4]                      #(8192,)
            mem_age = outputs_[5]                       #(8192,)
            recent_idx = outputs_[6]                    #(8192,)
            gradients = outputs_[7]                     #(16, 256)
            positive_gradient = outputs_[8]             #(16, 1)
            negative_gradient = outputs_[9]             #(16, 1)
            loss = outputs_[10]                         #(16, 1)
            incorrect_memory_lookup = outputs_[11]      #(16,)
            nearest_neighbor_gradient = outputs_[12]    #(16, 1)

            # mem_keys[fetched_idxs] == fetched_keys
            fetched_idxs = outputs_[13]                 #(16,)
            fetched_keys = outputs_[14]                 #(16,)
            fetched_vals = outputs_[15]                 #(16,)

            updated_fetched_keys = outputs_[16]         #(16, 128)
            updated_idxs = outputs_[17]                 #(16,)
            updated_keys = outputs_[18]                 #(16, 128)
            updated_vals = outputs_[19]                 #(16,)

            oldest_idxs = outputs_[20]                  #(16,)
            normalized_x = outputs_[21]                 #(16, 128)
            #y = outputs_[22]                            #

            mem_age_incr = outputs_[23]
            result = outputs_[24]

            mem_query_proj = outputs_[25]
            embedding = outputs_[26]

            neighbors_ypred_vals = outputs_[27]
            neighbors_ypred_sims = outputs_[28]
            sims_temp = outputs_[29]
            neighbors_ypred_idxs = outputs_[30]
            positive_gradient = outputs_[31]
            negative_gradient = outputs_[32]
            x = outputs_[33]
            x_pro = outputs_[34]
            normalized_x = outputs_[35]

            num_hits = outputs_[36]
            num_misses = outputs_[37]
            positive_gradient_mem_idx = outputs_[38]
            #diff_gradient = outputs_[39]
            #pos_bias = outputs_[39]
            #neg_bias = outputs_[40]
            grad_summ = outputs_[39]
            #positive_gradientss = outputs_[40]
            #largest_age = outputs_[41]
            #ipdb.set_trace()


            #last_positive_gradient_mem_idxs = outputs_[40]
            positive_gradients.append(positive_gradient)
            negative_gradients.append(negative_gradient)

            losses.append(loss_train)
            y_preds.append(y_pred_train)

            keys, vals, age = sess.run([self.mem_keys, self.mem_vals, self.mem_age])
            index = [i for i, ii in enumerate(keys) if np.any(keys[i]!=0.0)]

            precision = np.mean(yy==y_pred_train)

            #-----------------------------------------------------------------------------------------------------------
            #summaries:
            #ipdb.set_trace()
            summary = tf.Summary()
            summary.value.add(tag='Loss/avg_positive_gradient', simple_value=np.mean(positive_gradient))
            summary.value.add(tag='Loss/avg_negative_gradient', simple_value=np.mean(negative_gradient))
            #summary.value.add(tag='Loss/avg_diff_gradient', simple_value=np.mean(diff_gradient))
            summary.value.add(tag='Loss/sum_positive_gradient', simple_value=np.sum(positive_gradient))
            summary.value.add(tag='Loss/sum_negative_gradient', simple_value=np.sum(negative_gradient))
            summary.value.add(tag='Loss/loss', simple_value=np.mean(loss))
            summary.value.add(tag='Loss/loss_train', simple_value=loss_train)
            summary.value.add(tag='Neighborhood/sum(incorrect_memory_lookup)', simple_value=np.sum(incorrect_memory_lookup))
            summary.value.add(tag='Precision/precision_training', simple_value=precision)
            self.summary_writer.add_summary(summary, episode_count)

            self.summary_histogram(self.summary_writer, "Memory/mem_keys", np.ravel(mem_keys), episode_count)
            self.summary_histogram(self.summary_writer, "Memory/mem_vals", np.ravel(mem_vals), episode_count)
            self.summary_histogram(self.summary_writer, "Memory/mem_age", np.ravel(mem_age), episode_count)

            self.summary_histogram(self.summary_writer, "Gradients/positive_gradient", np.ravel(positive_gradient), episode_count)
            self.summary_histogram(self.summary_writer, "Gradients/negative_gradient", np.ravel(negative_gradient), episode_count)
            self.summary_histogram(self.summary_writer, "Gradients/incorrect_memory_lookup", np.ravel(incorrect_memory_lookup), episode_count)
            #self.log_histogram(self.summary_writer, "Gradients/pos_bias", np.ravel(pos_bias), episode_count)
            #self.log_histogram(self.summary_writer, "Gradients/neg_bias", np.ravel(neg_bias), episode_count)

            for grad_name in grad_summ:
                self.summary_histogram(self.summary_writer,
                                       "GradientsEncoder/%s"%grad_name,
                                       np.ravel(grad_summ[grad_name]),
                                       episode_count)

            self.summary_writer.flush()
            sess.run(self.increment_global_step)
            episode_count += 1
        memory_curr = sess.run([self.mem_keys, self.mem_vals, self.mem_age])
        logging.info('[TRAINING] Memory value distribution (after): ' + hist_str(memory_curr[1]))
        precision = np.mean(np.array(y) == np.array(y_preds))
        logging.info('[TRAINING] average loss: {:.4f} | precision in current episode: {:.4f}'.format(np.mean(losses), precision))
        logging.info('[TRAINING] episode length y: {} | batch size yy: {}'.format(len(y), len(yy)))
        logging.info('---')

        # neighborhood info for every training minibatch.
        # Note: this format assumes we send only an array of 1 minibatch
        neighborhood_info = {}
        neighborhood_info['neighbors_ypred_vals'] = neighbors_ypred_vals
        neighborhood_info['neighbors_ypred_sims'] = neighbors_ypred_sims
        neighborhood_info['neighbors_ypred_idxs'] = neighbors_ypred_idxs
        neighborhood_info['positive_gradient'] = positive_gradient
        neighborhood_info['negative_gradient'] = negative_gradient
        # neghborhood has changed after training step, so we cannot compare it
        #neighborhood_info['neigh_age'] = mem_age[neighbors_ypred_idxs]
        neighborhood_info['y'] = y
        self.neighborhood_str(neighborhood_info)

        return losses, y_preds

    def neighborhood_str(self, neighborhood_info):
        neighbors_ypred_vals = neighborhood_info['neighbors_ypred_vals']
        neighbors_ypred_sims = neighborhood_info['neighbors_ypred_sims']
        neighbors_ypred_idxs = neighborhood_info['neighbors_ypred_idxs']
        positive_gradient = neighborhood_info['positive_gradient']
        negative_gradient = neighborhood_info['negative_gradient']
        y = neighborhood_info['y'][0]

        output = {}
        output_str = ''
        for i in range(len(y)):
            neighbors = []
            for j in range(len(neighbors_ypred_vals[i])):
                val = neighbors_ypred_vals[i][j]
                sim = '{:.2}'.format(neighbors_ypred_sims[i][j])
                age = ''
                neighbors.append((val, sim))
            output[y[i]] = neighbors
            output_str +=  '{}: {}\n'.format(y[i], neighbors)
        logging.info('neighbors:\n{}'.format(output_str))

    def get_memory_state(self, sess):
        current_memory = sess.run([self.mem_keys, self.mem_vals, self.mem_age])
        return current_memory[0], current_memory[1], current_memory[2]

    def clear_memory(self, sess):
        sess.run(self.memory.clear_memory())

    def summary_histogram(self, writer, tag, values, step, bins=1000):
        # Convert to a numpy array
        values = np.array(values)

        # Create histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        writer.add_summary(summary, step)
        writer.flush()

    def step_validation(self, sess, x, x_words, y, clear_memory=False):
        '''
        Inference knowing value of label y
        :param sess: current session
        :param x: input episode (list of mini-batches)
        :param clear_memory: to clear memories
        :return: list of predicted values
        '''

        if clear_memory:
            sess.run([self.memory.clear_memory()])

        predictions = []
        hits = []

        ops = [self.memory.y_pred,
               self.memory.neighbors_ypred_sims,
               self.memory.neighbors_ypred_idxs,
               self.memory.neighbors_ypred_vals,
               self.memory.mem_vals,
               #self.y_pred_train,

               self.memory.mem_keys,
               self.memory.mem_age]


        for xx, xx_words, yy in zip(x, x_words, y):
            xx = xx.reshape([-1, len(xx)]) if xx.ndim == 1 else xx
            xx_words = xx_words.reshape([-1, len(xx_words)]) if xx_words.ndim == 1 else xx_words
            yy = [yy] if yy.ndim == 0 else yy

            ops_ = sess.run(ops, feed_dict={self.x: xx[:, :self.seq_len_chars],
                                            self.x_words: xx_words[:, :self.seq_len_words],
                                            self.y: yy,
                                            self.encoder.seqlen: np.zeros(len(xx))})

            y_pred = ops_[0]
            mem_ypred_sims = ops_[1]
            mem_ypred_idx = ops_[2]
            mem_ypred_values = ops_[3]
            mem_vals = ops_[4]
            #y_pred_eval = ops_[5]

            predictions.append(y_pred)
            hits.append(y_pred == yy)

        # -----------------------------------------------------------------------------------------------------------
        # Summary:
        episode_count = sess.run(self.global_step)
        precision = np.mean(hits)

        summary = tf.Summary()
        summary.value.add(tag='Precision/precision_validation', simple_value=precision)
        self.summary_writer.add_summary(summary, episode_count)
        self.summary_writer.flush()

        #mem_vals[mem_ypred_idx_val] == y_pred
        #y_pred = np.array(Counter(mem_ypred_values[0]).most_common(1)[0][0])
        return predictions

    def step_validation_info(self, sess, x, x_words, y, clear_memory=False):
        '''
        Inference knowing value of label y
        :param sess: current session
        :param x: input episode (list of mini-batches)
        :param clear_memory: to clear memories
        :return: list of predicted values
        '''

        if clear_memory:
            sess.run([self.memory.clear_memory()])

        predictions = []
        hits = []
        sims = []

        ops = [self.memory.y_pred,
               self.memory.neighbors_ypred_sims,
               self.memory.neighbors_ypred_idxs,
               self.memory.neighbors_ypred_vals,
               self.memory.mem_vals,
               #self.y_pred_train,

               self.memory.mem_keys,
               self.memory.mem_age]


        for xx, xx_words, yy in zip(x, x_words, y):
            xx = xx.reshape([-1, len(xx)]) if xx.ndim == 1 else xx
            xx_words = xx_words.reshape([-1, len(xx_words)]) if xx_words.ndim == 1 else xx_words
            yy = [yy] if yy.ndim == 0 else yy

            ipdb.set_trace()
            ops_ = sess.run(ops, feed_dict={self.x: xx[:, :self.seq_len_chars],
                                            self.x_words: xx_words[:, :self.seq_len_words],
                                            self.y: yy,
                                            self.encoder.seqlen: np.zeros(len(xx))})

            y_pred = ops_[0]
            mem_ypred_sims = ops_[1]
            mem_ypred_idx = ops_[2]
            mem_ypred_values = ops_[3]
            mem_vals = ops_[4]
            #y_pred_eval = ops_[5]

            predictions.append(y_pred)
            hits.append(y_pred == yy)
            sims.append(mem_ypred_sims)

        # -----------------------------------------------------------------------------------------------------------
        # Summary:
        episode_count = sess.run(self.global_step)
        precision = np.mean(hits)

        summary = tf.Summary()
        summary.value.add(tag='Precision/precision_validation', simple_value=precision)
        self.summary_writer.add_summary(summary, episode_count)
        self.summary_writer.flush()

        #mem_vals[mem_ypred_idx_val] == y_pred
        #y_pred = np.array(Counter(mem_ypred_values[0]).most_common(1)[0][0])
        output = {'predictions': predictions, 'sims': sims}
        return output

    def step_inference(self, sess, x, y, clear_memory=False):
        '''
        Inference without knowing value of label y
        :param sess: current session
        :param x: input episode (list of mini-batches)
        :param clear_memory: to clear memories
        :return: list of predicted values
        '''

        if clear_memory:
            sess.run([self.memory.clear_memory()])

        y_preds = []
        ops = [self.memory.y_pred,
               self.memory.neighbors_ypred_sims,
               self.memory.neighbors_ypred_idxs,
               self.memory.neighbors_ypred_vals,
               self.memory.mem_vals,
               self.memory.mem_keys,
               self.memory.mem_age]
        for xx in x:
            ops_ = sess.run(ops, feed_dict={self.x: xx[:, :self.seq_len_chars],
                                            self.encoder.seqlen: np.zeros(len(xx))})

            y_pred = ops_[0]
            mem_ypred_sims = ops_[1]
            mem_ypred_idx = ops_[2]
            mem_ypred_values = ops_[3]
            mem_vals = ops_[4]

            y_preds.append(y_pred)

        average_precision = np.mean(np.array(y_preds) == np.array(y))

        summary = tf.Summary()
        summary.value.add(tag='Precision/precision_validation', simple_value=average_precision)
        episode_count = sess.run(self.global_step)
        self.summary_writer.add_summary(summary, episode_count)
        self.summary_writer.flush()

        #y_pred = np.array(Counter(mem_ypred_values[0]).most_common(1)[0][0])
        return y_preds

    def step_key_stats(self, sess, x, x_words, clear_memory=False):
        '''
        Key-based analysis of inference without knowing value of label y
        :param sess: current session
        :param x: input episode (list of mini-batches)
        :param clear_memory: to clear memories
        :return: list of predicted values
        '''

        if clear_memory:
            sess.run([self.memory.clear_memory()])

        ypreds, key_idxs, age, keys, sims  = [], [], [], [], []
        ops = [self.memory.y_pred,
               self.memory.neighbors_ypred_sims,
               self.memory.neighbors_ypred_idxs,
               self.memory.neighbors_ypred_vals,
               self.memory.mem_keys,
               self.memory.mem_vals,
               self.memory.mem_age]
        for xx, xx_words in zip(x, x_words):
            # when iterating one observation per each step
            xx = xx.reshape([-1, len(xx)]) if xx.ndim == 1 else xx
            xx_words = xx_words.reshape([-1, len(xx_words)]) if xx_words.ndim == 1 else xx_words

            ops_ = sess.run(ops, feed_dict={self.x: xx[:, :self.seq_len_chars], self.x_words: xx_words[:, :self.seq_len_words], self.encoder.seqlen: np.zeros(len(xx))})

            y_pred = ops_[0]
            mem_ypred_sims = ops_[1]
            mem_ypred_idx = ops_[2]
            mem_ypred_values = ops_[3]

            mem_keys = ops_[4]
            mem_vals = ops_[5]
            mem_age = ops_[6]

            key_idx = mem_ypred_idx[0][0]
            key_idxs.append(key_idx)
            age.append(mem_age[key_idx])
            keys.append(mem_keys[key_idx])
            ypreds.append(mem_vals[key_idx])        #ypreds.append(y_pred)
            sims.append(mem_ypred_sims[0][0])


            #mem_vals[mem_ypred_idx_val] == y_pred
            #y_pred = np.array(Counter(mem_ypred_values[0]).most_common(1)[0][0])
        return ypreds, key_idxs, keys, age, sims

    def step_inference_neighborsidx(self, sess, x, clear_memory=False, debug=False):
        '''
        Inference without knowing value of label y
        :param sess: current session
        :param x: input episode (list of mini-batches)
        :param clear_memory: to clear memories
        :return: list of predicted values
        '''

        if clear_memory:
            sess.run([self.memory.clear_memory()])

        predictions = []
        neighs_sims = []
        neighbors_vals = []

        # x
        # [matrix([[11.0943606, 14.9022914]]),
        #  matrix([[15.87570672, 18.01953229]]),
        #  matrix([[16.22724095, 11.52977574]]),
        #  matrix([[19.34007288, 1.27217744]]),
        #  matrix([[7.37841634, -1.90224113]]),
        for xx in x:
            output, neighbors_sims, neighbors_mem_vals = sess.run([self.memory.y_pred_val,  ###self.y_test,
                                                                   self.memory.neigh_ypred_sims_val,
                                                                   self.memory.mem_ypred_keys_val],
                                                                  feed_dict={
                                                                 self.x: np.matrix(xx)[:, :self.seq_len_chars],
                                                                 self.encoder.seqlen: np.zeros(len(xx))
                                                             })

            #len(neighbors_mem_vals[0]): 160
            prediction = np.array(Counter(neighbors_mem_vals[0]).most_common(1)[0][0])

            #majority voting
            # prediction_sets = []
            # for table_id in range(len(neigh_idx)):
            #     preds_table = sess.run(self.memories.access(neigh_idx[table_id]))
            #     prediction_sets.append(set(preds_table.ravel()))
            #prediction_voting = Counter([ee for e in prediction_sets for ee in e]).most_common(1)[0][0]
            #prediction = prediction_voting

            #ipdb.set_trace()
            predictions.append(prediction)
            neighs_sims.append(neighbors_sims)
            neighbors_vals.append(neighbors_mem_vals)
            #print('xx: ', xx.shape)
        return predictions


    def step_validation2(self, sess, x, y, clear_memory=False):
        '''
        crunching numbers over batches: put training complexity here
        very batch is: (batch_size ,input_dim)
        :param sess:
        :param x: A list of batches of observations defining the episode
        :param y: A list of batches of labels corresponding to x
        :param clear_memory: whether to clear the memories before an episode
        :return:
        '''

        logging.info('----------------------------------------------------------')
        logging.info('[VALIDATION] label distribution in episode:\t' + hist_str(y))

        #keep current state of memories
        curr_memory_before = sess.run([self.mem_keys, self.mem_vals, self.mem_age])
        if clear_memory:
            sess.run(self.memory.clear_memory())
        curr_memory_after = sess.run([self.mem_keys, self.mem_vals, self.mem_age])

        #output = [self.y_val]
        outputs = [
            self.y_test,                             # 0

            self.memory.neighbor_nearest_idx_val,
            self.memory.nearest_nolabel_idx_temp_k_blind,
            self.memory.mem_ypred_idx_val,
            self.memory.mem_ypred_idxs_val,

            self.memory.mem_vals,
            self.memory.mem_ypred_keys_val,
            self.memory.mem_keys,
            self.memory.batch_size_blind,
            self.memory.neigh_ypred_sims_val,
            self.memory.mem_ypred_keys_val,

            self.memory.mem_query_proj
        ]

        self.query_ypreds = []
        self.query_neigh_distances = []
        self.query_neigh_keys = []
        for xx, yy in zip(x, y):
            #output_ = sess.run(output, feed_dict = {self.x: xx, self.y: yy})
            # output_ = sess.run(outputs, feed_dict = {self.x: xx,
            #                                          self.y: yy,
            #                                          self.embedder.seqlen: np.zeros(len(xx))
            #                                          })   #xx: (1, 784)   yy:(1,)

            output_ = sess.run(outputs, feed_dict={self.x: xx[:, :self.seq_len_chars],#.astype(int),
                                                   #self.y: yy,
                                                   self.encoder.seqlen: np.zeros(len(xx)),
                                                    #self.embedder.batch_size: len(xx)
                                                   })

            y_pred = output_[0]
            nearest_nolabel_idx_temp = output_[1]
            nearest_nolabel_idx_temp_k = output_[2]
            nearest_nolabel_idx_val = output_[3]
            neighbors_idxs_val = output_[4]         #<--

            mem_vals = output_[5]
            neighbors_mem_vals_val = output_[6]     #<-- similar classes in decreasing order
            mem_keys = output_[7]
            batch_size_val = output_[8]

            sims_temp_val = output_[9]             #<--
            neighbors_mem_keys_val = output_[10]    #<--
            mem_query_proj = output_[11]

            self.query_ypreds.append(y_pred)
            #ipdb.set_trace()
            logging.info('\t\ty: %s predicted: %s similarity: %s' %(str(yy),
                                                                    str(neighbors_mem_vals_val),
                                                                    str(neighbors_similarities_val)))

            #logging.info('label actual: %d predicted: %d' %(yy[0], self.y_pred[0]))
            #logging.info('self.fetched_idxs_: ', self.fetched_idxs_)
            #logging.info('curr_memory[0][self.fetched_idxs_]: ', curr_memory[0][self.fetched_idxs_])
            #logging.info('self.neighbors_mem_vals_: ', self.neighbors_mem_vals_)


        curr_memory_curr = sess.run([self.mem_keys, self.mem_vals, self.mem_age])

        logging.info('[VAL] Memory value distribution (before):\t\t' + hist_str(curr_memory_before[1]))
        logging.info('[VAL] Memory value distribution (after):\t\t' + hist_str(curr_memory_curr[1]))
        logging.info('[VAL] Predicted labels:\t\t\t\t\t\t' + hist_str(self.query_ypreds))
        comparison = ['%dy->%d'%(y[i][0], self.query_ypreds[i][0]) for i in range(len(y))]
        comparison_sims = ['%dy->%d' % (y[i][0], self.query_ypreds[i][0]) for i in range(len(y))]
        logging.info('[VAL] comparison: (actual -> predicted)\t\t' + str(comparison))
        logging.info('---')
        #ipdb.set_trace()

        #restore original state of memories before inference:
        #   - insert new label
        #   - modify existing key
        #sess.run(self.mem_reset_op, feed_dict={self.mem_keys_reset: curr_memory_before[0],
        #                                       self.mem_vals_reset: curr_memory_before[1],
        #                                       self.mem_age_reset: curr_memory_before[2]})
        return self.query_ypreds



