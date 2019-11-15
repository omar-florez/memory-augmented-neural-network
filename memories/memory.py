'''
Memory definition and operations
'''

import numpy as np
import tensorflow as tf
import ipdb

tf.set_random_seed(0)

class Memory(object):
    def __init__(self, key_dim, memory_size, choose_k=256, alpha=1.0, correct_in_top=1, age_noise=8.0,
                 var_cache_device='', nn_device='', keys_are_trainable=False, reg_coeff = 0.01, continous=True):
        self.key_dim = key_dim
        self.memory_size = memory_size

        #to limit the size of neighborhood to a smaller size of similar elements
        self.choose_k = choose_k
        self.alpha = alpha
        self.reg_coeff = reg_coeff
        self.correct_in_top = correct_in_top
        self.age_noise = age_noise
        #to cache variables??
        self.var_cache_device = var_cache_device
        #device in which to perform NN query (GPU, CPU)
        self.nn_device = nn_device

        caching_device = var_cache_device
        self.update_memory = tf.constant(True)

        self.mem_keys = tf.get_variable('keys', [self.memory_size, self.key_dim],
                                      initializer = tf.random_uniform_initializer(-0.0, 0.001),
                                      caching_device=caching_device, trainable=True)
        self.mem_vals = tf.get_variable('vals', [self.memory_size], dtype = tf.int32,
                                        initializer = tf.constant_initializer(0, tf.int32),
                                        caching_device = caching_device, trainable=False)
        self.mem_age = tf.get_variable('age', [self.memory_size], dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.0),
                                       caching_device = caching_device, trainable=False)
        self.std = tf.get_variable('std', [self.memory_size], dtype=tf.float32,
                                   caching_device=caching_device, trainable=True,
                                   initializer=tf.constant_initializer(0.0))
        self.recent_idx = tf.get_variable('recent_idx', [self.memory_size], dtype=tf.int32,
                                          initializer=tf.constant_initializer(0, tf.int32), trainable=False)

        self.mem_query_proj = tf.get_variable('query_proj', [self.key_dim, self.key_dim],
                                              dtype=tf.float32, trainable=False,
                                              initializer=tf.random_uniform_initializer(0.0, 0.001),
                                              caching_device = caching_device)



    def query(self, x, y, use_recent_idx=False, use_centroid=False):
        '''
        responsible for generating loss function based on nearest neighbors
        :param x: query embedding (batch_size, key_dims)
        :param y: query label (batch_size,)
        :return: (result, mask, teacher_loss)
            y_pred: result of memories look up
            mask: affinity of query to the result
            loss: average training loss
        '''

        self.normalized_x = self.project_query(x)

        #x_pro = (batch_size, key_dim)
        if use_centroid:
            x_pro = self.get_x_prototype(x, y)
        else:
            x_pro = x

        with tf.name_scope('vector_lookup'):
            #neighbors_ypred_sims: contains sims not sorted
            #y_pred                 = (?,)
            #neighbors_ypred_idxs   = (?, choose_k)
            #neighbors_ypred_sims   = (?, choose_k)
            #neighbors_ypred_vals = (?, choose_k)
            y_pred, neighbors_ypred_idxs, neighbors_ypred_sims, neighbors_ypred_vals = self.query_lookup(x_pro)
            self.y_pred = y_pred

        batch_size = tf.shape(x)[0]

        #---------------------------------------------------------------------------------------------------------------
        # Gradient computation:


        # gradients =   array([[0., 0., 1., ..., 0., 0., 0.],
        #                     [0., 0., 0., ..., 1., 0., 0.],
        #                     [1., 0., 0., ..., 0., 0., 0.],
        #                     ...,
        #                     [0., 0., 0., ..., 0., 0., 0.],
        #                     [1., 1., 0., ..., 0., 0., 0.],
        #                     [0., 0., 0., ..., 0., 0., 0.]]
        # gradients = (batch_size, choose_k) = (batch_size, choose_k) - (batch_size, 1)
        #   convert gradients to 1s (hits) and 0s (misses) in neighborhood
        gradients = tf.to_float(tf.abs(neighbors_ypred_vals - tf.expand_dims(y, 1)))
        gradients = 1.0 - tf.minimum(1.0, gradients)
        ##gradients = gradients*2.0-1.0               # to set hit:+1 and miss:-1

        # num_hits = num_misses = (?,)
        num_hits = tf.cast(tf.reduce_sum(gradients, axis=1), dtype=tf.int32)
        num_misses = tf.cast(tf.reduce_sum(1.0-gradients, axis=1), dtype=tf.int32)

        #get top-num_hits hit and miss candidates after multiplying (gradient x similarity)
        #       positive_gradient = (batch_size, 1) = top_k((batch_size, choose_k)*(batch_size, choose_k), 1)
        positive_gradient, positive_gradient_neigh_idx = tf.nn.top_k(gradients*neighbors_ypred_sims,
                                                                           k=self.correct_in_top)         #<--
        negative_gradient, negative_gradient_neighborhoodidx = tf.nn.top_k((1.0-gradients)*neighbors_ypred_sims,
                                                                           k=self.correct_in_top)   #<--

        # Multiply each gradient with 0/1 to remove invalid gradients with all entries equal to zero:
        #       1 if valid entry        non-zero gradient, right label in the neighborhood): hit
        #       0 if invalid gradient   when all neighbors are misses (incorrect label), so *=(1.0-1.0)
        positive_gradient *= tf.expand_dims((1.0 - tf.to_float(tf.equal(0.0, tf.reduce_sum(gradients, axis=1)))), 1)
        #to set -1 reward when no label found
        positive_gradient -= tf.expand_dims((tf.to_float(tf.equal(0.0, tf.reduce_sum(gradients, axis=1)))), 1)

        # pos_bias = tf.get_variable('pos_bias', [128], dtype=tf.float32, initializer=tf.constant_initializer(0.0, tf.float32))
        # neg_bias = tf.get_variable('neg_bias', [128], dtype=tf.float32, initializer=tf.constant_initializer(0.0, tf.float32))
        # positive_gradient +=  pos_bias
        # negative_gradient +=  neg_bias

        # self.mem_vals = tf.get_variable('memvals', [self.memory_size], dtype = tf.int32,
        #                                 initializer = tf.constant_initializer(0, tf.int32),
        #                                 caching_device = caching_device, trainable=False)

        #---------------------------------------------------------------------------------------------------------------
        # loss definition

        #(batch_size,1)
        # TODO: check loss definition. Is better: cross entropy, negative contrastive estimation
        #loss = tf.nn.relu(negative_gradient-positive_gradient+self.alpha)-self.alpha
        #0.9
        ###loss = tf.nn.relu(tf.reduce_sum(negative_gradient) - tf.reduce_sum(positive_gradient) + self.alpha) - self.alpha

        #loss = negative_gradient/tf.to_float(tf.shape(negative_gradient)[0]) - positive_gradient

        # LOSS:7 0.7161
        #loss = tf.maximum(negative_gradient - positive_gradient+0.2, 0.0)
        # LOSS:7a 0.65
        #loss = tf.minimum(negative_gradient - positive_gradient, 0.0)
        # LOSS:7b
        #loss = tf.minimum(negative_gradient - positive_gradient+0.2, 0.0)

        #loss = tf.nn.relu(negative_gradient - positive_gradient)
        #loss = negative_gradient - positive_gradient
        #loss = tf.minimum(negative_gradient - positive_gradient, self.alpha)

        # LOSS:6 0.7104
        #loss = tf.tanh(negative_gradient - positive_gradient + self.alpha) - self.alpha

        ##loss = tf.nn.relu(negative_gradient - positive_gradient + self.alpha) - self.alpha

        #------------------------
        # Original:
        loss = tf.nn.relu(negative_gradient - positive_gradient + 1.0) - 1.0 + 8.0*tf.reduce_sum(1.0-gradients)

        #loss = tf.reduce_sum((1.0 - gradients) * neighbors_ypred_sims)
        #loss = tf.reduce_sum(gradients * neighbors_ypred_sims)

        #------------------------
        #loss2:
        #loss = 4.0*tf.reduce_mean((1.0-gradients)*neighbors_ypred_sims)-tf.reduce_mean(gradients*neighbors_ypred_sims)
        ##loss = -tf.reduce_mean((gradients) * neighbors_ypred_sims, 1)
        #loss = tf.reduce_sum((gradients) * neighbors_ypred_sims, 1)

        #loss = 4.0 * tf.reduce_mean(positive_gradient) - tf.reduce_mean(negative_gradient)

        #0.8642857142857143
        #loss = tf.exp(tf.reduce_sum((1.0 - gradients) * neighbors_ypred_sims) - tf.reduce_sum(gradients * neighbors_ypred_sims)) + 8.0*tf.reduce_sum(1.0-gradients)

        ## LOSS:3 Exponential: 0.45
        #loss = tf.reduce_sum(tf.exp((1.0 - gradients) * neighbors_ypred_sims)) - tf.reduce_sum(tf.exp(-(gradients * neighbors_ypred_sims)))

        #LOSS:5
        #loss = tf.reduce_sum((1.0 - gradients) * neighbors_ypred_sims) - tf.reduce_sum(gradients * neighbors_ypred_sims) + 8.0*tf.reduce_sum(1.0-gradients)
        #------------------------

        #neighbors_ypred_sims:    (?, choose_k)
        #gradients:                 (?, choose_k)
        #loss = tf.reduce_sum((1.0 - gradients) * neighbors_ypred_sims) - tf.reduce_sum(gradients * neighbors_ypred_sims) + 8.0*tf.reduce_sum(1.0 - gradients)

        #LOSS:4 multiply negative as regularization: 0.9035714285714286
        #loss = tf.reduce_sum((1.0 - gradients) * neighbors_ypred_sims) - tf.reduce_sum(gradients * neighbors_ypred_sims) + tf.reduce_prod((1.0 - gradients) * neighbors_ypred_sims)

        # LOSS:5
        #loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01


        #positive_gradient_neigh_idx = (?, 1) -> (?,)
        #           tf.reshape(neighbors_ypred_idxs, [-1])                 (?, choose_k) -> (?,batch*choose_k)
        self.index_pos = positive_gradient_neigh_idx[:, 0] + self.choose_k * tf.range(batch_size)
        positive_gradient_mem_idx = tf.gather(tf.reshape(neighbors_ypred_idxs, [-1]), self.index_pos) #shape=(?,)

        self.index_neg = negative_gradient_neighborhoodidx[:, 0] + self.choose_k * tf.range(batch_size)
        negative_gradient_mem_idx = tf.gather(tf.reshape(neighbors_ypred_idxs, [-1]), self.index_neg)  # shape=(?,)

        pos_keys = tf.gather(self.mem_keys, positive_gradient_mem_idx)
        neg_keys = tf.gather(self.mem_keys, negative_gradient_mem_idx)

        #---------------------------------------------------------------------------------------------------------------
        # collect correct fetched positions in memories to perform update
        #       - first column of 'gradients' is the most similar instance given similarity matrix
        #       - In which each row is ordered in decreasing order.
        #       - If first column contains zero gradients, it means the most similar element was not recognized,
        #       hence a incorrect memories lookup
        #(batch_size,1)
        nearest_neighbor_gradient = tf.slice(gradients, [0,0], [-1, self.correct_in_top])
        # (batch_size,)
        incorrect_memory_lookup = tf.equal(0.0, tf.reduce_sum(nearest_neighbor_gradient, 1))

        # ---------------------------------------------------------------------------------------------------------------
        # Correct loss functions:
        regularizer = self.reg_coeff * tf.reduce_sum(tf.cast(incorrect_memory_lookup, tf.float32))
        diff_gradient = tf.reduce_sum(pos_keys * neg_keys, axis=1)

        ##LTH-loss1: 0.71
        #loss = tf.nn.relu(negative_gradient - positive_gradient + self.alpha) #- self.alpha + regularizer
        #loss = loss + regularizer

        #loss = tf.nn.relu(negative_gradient - positive_gradient + self.alpha) - self.alpha + tf.reduce_sum(
        #    tf.cast(incorrect_memory_lookup, tf.float32))

        # loss inspired in Word2Vec: similar results to ReLU loss
        #loss = -tf.log(tf.nn.sigmoid(positive_gradient)) + tf.log(tf.nn.sigmoid(negative_gradient))

        #loss = tf.nn.relu(negative_gradient - positive_gradient + tf.expand_dims(diff_gradient, 1) + self.alpha) - self.alpha
        #loss = tf.nn.relu(negative_gradient + positive_gradient + tf.expand_dims(diff_gradient, 1) + self.alpha) - self.alpha

        with tf.device(self.var_cache_device):
            #fetched indices are those that showed the best correlation between (gradient*similarity)
            #gradient isn 1 if it's a top-1 hit
            fetched_idxs = positive_gradient_mem_idx
            fetched_keys = tf.gather(self.mem_keys, fetched_idxs)
            fetched_vals = tf.gather(self.mem_vals, fetched_idxs)

        #---------------------------------------------------------------------------------------------------------------
        #implement update operations of keys, val, and age:
        updated_fetched_keys = tf.nn.l2_normalize(self.normalized_x + fetched_keys, dim=1)
        mem_age_with_noise = self.mem_age + tf.random_uniform([self.memory_size], -self.age_noise, self.age_noise)
        _, oldest_idxs = tf.nn.top_k(mem_age_with_noise, k=batch_size, sorted=False)

        #tf.where(condition, x=None, y=None, name=None)
        #if incorrect key lookup: (maybe doesnt exist in neighborhood)
        #   insert l2_normalize(original query)
        # else
        #   l2_normalize(original query + most similar keys in memories)
        # TODO: new ages, new keys, new vals
        # with tf.control_dependencies([y_pred]):
        #     ipdb.set_trace()
        #     updated_idxs = tf.where(incorrect_memory_lookup, oldest_idxs, fetched_idxs)
        #     updated_keys = tf.where(incorrect_memory_lookup, self.normalized_x, updated_fetched_keys)
        #     updated_vals = tf.where(incorrect_memory_lookup, y, fetched_vals)

        with tf.control_dependencies([y_pred]):
            updated_idxs = tf.where(incorrect_memory_lookup, oldest_idxs, fetched_idxs)
            ##updated_idxs = tf.where(incorrect_memory_lookup, fetched_idxs, fetched_idxs)
            updated_keys = tf.where(incorrect_memory_lookup, self.normalized_x, updated_fetched_keys)
            updated_vals = tf.where(incorrect_memory_lookup, y, fetched_vals)

        # ---------------------------------------------------------------------------------------------------------------
        #perform updates:
        #Applies sparse updates to a variable reference:
        # scatter_update(ref, indices, updates) = ref[indices, ...] = updates[...]
        #update ages of all memories entries and refresh those one fetched
        mem_age_incr = self.mem_age.assign_add(tf.ones([self.memory_size], dtype=tf.float32))

        a = tf.scatter_update(self.mem_age, updated_idxs, tf.zeros([batch_size], dtype=tf.float32))
        b = tf.scatter_update(self.mem_keys, updated_idxs, updated_keys)
        c = tf.scatter_update(self.mem_vals, updated_idxs, updated_vals)

        ## make similar and negative class more different
        ##negative_keys = tf.gather(self.mem_keys, negative_gradient_mem_idx)
        ##negative_updated_fetched_keys = tf.nn.l2_normalize(self.normalized_x - negative_keys, dim=1)
        ##nn = tf.scatter_update(self.mem_keys, negative_gradient_mem_idx, negative_updated_fetched_keys)

        if use_recent_idx:
            d = tf.scatter_update(self.recent_idx, y, updated_idxs)
        else:
            d = tf.group()

        #---------------------------------------------------------------------------------------------------------------
        # LSH:

        #A list of `Operation` or `Tensor` objects which must be executed or computed before running the operations
        # defined in the context.
        with tf.control_dependencies([tf.group(a, b, c, d, mem_age_incr)]):
            y_pred_train = tf.identity(y_pred)
            #mask = tf.identity(mask)
            loss = tf.identity(loss)

        #--------------------------------------------------------------------------------------
        self.neighbors_ypred_idxs = neighbors_ypred_idxs
        self.positive_gradient_neigh_idx = positive_gradient_neigh_idx

        self.gradients = gradients                                          #(16, 256)
        self.positive_gradient = positive_gradient                          #(16, 1)
        self.negative_gradient = negative_gradient                          #(16, 1)
        self.diff_gradient = diff_gradient
        self.loss = loss
        self.incorrect_memory_lookup = incorrect_memory_lookup
        self.nearest_neighbor_gradient = nearest_neighbor_gradient          #(16, 1)

        self.updated_idxs = updated_idxs
        self.updated_keys = updated_keys
        self.updated_vals = updated_vals

        self.oldest_idxs = oldest_idxs
        self.normalized_x_val = self.normalized_x
        self.x = x
        self.y = y
        self.x_pro = x_pro

        self.fetched_keys = fetched_keys
        self.fetched_idxs = fetched_idxs
        self.updated_fetched_keys = updated_fetched_keys
        self.fetched_vals = fetched_vals

        self.mem_age_incr = mem_age_incr
        #self.result = y_pred
        #self.mask = mask                                                    #(16, 255)

        self.neighbors_ypred_vals = neighbors_ypred_vals
        self.neighbors_ypred_sims = neighbors_ypred_sims
        self.neighbors_ypred_idxs = neighbors_ypred_idxs
        self.sims_temp = neighbors_ypred_sims

        self.num_hits = num_hits
        self.num_misses = num_misses
        self.positive_gradient_mem_idx = positive_gradient_mem_idx
        #self.regularizer = regularizer
        #self.pos_bias = pos_bias
        #self.neg_bias = neg_bias

        return y_pred_train, tf.reduce_mean(loss)

    def get_x_prototype(self, x, y):
        '''
        responsible for generating initial gradients based on nearest neighbors
        :param x: query (batch_size, key_dims)
        :return: (result, mask, teacher_loss)
            result: result of memories look up
            mask: affinity of query to the result
        '''
        batch_size = tf.shape(x)[0]


        #lookup query vector using the memories index
        #       neighbors_ypred_idxs: [batch_size, num_per_bucket]
        neighbors_ypred_sims, neighbors_ypred_idxs = self.get_nearest_neighbor_idxs(self.normalized_x)
        neighbors_ypred_sims = tf.squeeze(neighbors_ypred_sims, [1], 'neighborhood_sims')

        #choose_k == num_per_bucket
        choose_k = tf.shape(neighbors_ypred_idxs)[1]

        #compute similarities in the smaller neighborhood and result
        with tf.device(self.var_cache_device):
            #(batch_size, 257, dim_key) = tf.gather((mem_size, dim_key), (batch_size, 256))
            mem_ypred_keys = tf.stop_gradient(tf.gather(self.mem_keys, neighbors_ypred_idxs, name='query_mem_keys_val'))
            # (batch_size, 257 neighbors)
            neighbors_ypred_values = tf.gather(self.mem_vals, neighbors_ypred_idxs, name='hint_pool_mem_val')
            neighbors_ypred_keys = tf.gather(self.mem_keys, neighbors_ypred_idxs)

            #(16,1): the most similar element within the neighborhood
            neighbor_nearest_idx = tf.to_int32(tf.arg_max(neighbors_ypred_sims[:,:choose_k], 1))
            mem_ypred_idx = tf.gather(tf.reshape(neighbors_ypred_idxs, [-1]), neighbor_nearest_idx + choose_k*tf.range(batch_size))

            #(batch_size,)
            #result is based on most similar keys (first column in sorted neighborhood) excluding the recent elements
            y_pred = tf.gather(self.mem_vals, tf.reshape(mem_ypred_idx, [-1]))

        ##############
        gradients = tf.to_float(tf.abs(neighbors_ypred_values - tf.expand_dims(y, 1)))
        # convert gradients to 0s (miss in neighborhood) and 1s (hit in neighborhood)
        # gradients: (?, 50)
        gradients = 1.0 - tf.minimum(1.0, gradients)
        #hit_keys: (?, key_dim)
        hit_keys = tf.reduce_sum(neighbors_ypred_keys * tf.expand_dims(gradients, 2), axis=1) / (tf.reduce_sum(gradients) + 0.0001)
        return hit_keys

    def query_lookup(self, x_query):
        '''
        responsible for generating initial gradients based on nearest neighbors
        :param x: query (batch_size, key_dims)
        :return: (result, mask, teacher_loss)
            result: result of memories look up
            mask: affinity of query to the result
        '''
        batch_size = tf.shape(x_query)[0]

        #lookup query vector using the memories index
        #       neighbors_ypred_idxs: [batch_size, num_per_bucket*num_tables]
        #       neighbors_ypred_sims: [batch_size, num_per_bucket*num_tables] -> [batch_size, num_per_bucket*num_tables]
        neighbors_ypred_sims, neighbors_ypred_idxs = self.get_nearest_neighbor_idxs(self.project_query(x_query))

        #choose_k == num_per_bucket
        ##choose_k = tf.shape(neighbors_ypred_idxs)[1]
        ##choose_k = self.choose_k

        #compute similarities in the smaller neighborhood and result
        with tf.device(self.var_cache_device):
            #(batch_size, 257, dim_key) = tf.gather((mem_size, dim_key), (batch_size, 256))
            mem_ypred_keys = tf.gather(self.mem_keys, neighbors_ypred_idxs, name='query_mem_keys_val')
            # (batch_size, 257 neighbors)
            neighbors_ypred_values = tf.gather(self.mem_vals, neighbors_ypred_idxs, name='hint_pool_mem_val')

            #(16,1): the most similar element within the neighborhood
            neighbor_nearest_idx = tf.to_int32(tf.arg_max(neighbors_ypred_sims, 1))

            #(16,): getting the first column of [[neighbors, ...]] (first element in each neighboring array)
            #get the index with the highest similarity for each observation in the minibatch
            #       - neighbor_nearest_idx: contains the relative position of the nearest key for each observation
            #       - choose_k*tf.range(batch_size): used to offset the relative position every within the 'choose_k' candidates
            mem_ypred_idx = tf.gather(tf.reshape(neighbors_ypred_idxs, [-1]), neighbor_nearest_idx + self.choose_k*tf.range(batch_size))

            #(batch_size,)
            #result is based on most similar keys (first column in sorted neighborhood) excluding the recent elements
            y_pred = tf.gather(self.mem_vals, tf.reshape(mem_ypred_idx, [-1]))

        #create mask
        # TODO: create attention mask here?
        #reduce temperature by 20% of all neighbors but the current instance
        #softmax_temperature = max(1.0, np.log(0.2*self.choose_k)/self.alpha)
        #mask =  tf.nn.softmax(softmax_temperature*neighbors_ypred_sims[:,:self.choose_k-1])
        #neighbors_ypred_sims: (?, num_tables*choose_k) = (?, 5*50)
        #mask = tf.nn.softmax(neighbors_ypred_sims[:, :self.choose_k])


        #with tf.name_scope('memories.blind_query'):
        #    tf.summary.histogram('mem_query_proj', self.mem_query_proj)     #(128, 128)

        # self.y_pred = y_pred
        # self.neighbors_ypred_idxs = neighbors_ypred_idxs
        # self.neighbors_ypred_sims = neighbors_ypred_sims
        # self.neighbors_ypred_vals = neighbors_ypred_values

        return y_pred, neighbors_ypred_idxs, neighbors_ypred_sims, neighbors_ypred_values

    def access(self, indices):
        return tf.gather(self.mem_vals, indices)

    def project_query(self, x):
        # #Note: np.sum(normalized_x[0]**2)==1

        x = tf.matmul(x, self.mem_query_proj)  # (16, 128)x(128, 128)
        normalized_x = tf.nn.l2_normalize(x, dim=1)

        #(batch_size, key_dim)x(key_dim, key_dim)
        # normalized_x = tf.nn.l2_normalize(x, dim=1)
        #normalized_x = tf.nn.sigmoid(tf.matmul(normalized_x, self.mem_query_proj))
        #normalized_x = tf.nn.l2_normalize(normalized_x, dim=1)
        return normalized_x

    #-------------------------------------------------------------------------------------------------------------------
    # Nearest Neighbor methods:
    #-------------------------------------------------------------------------------------------------------------------

    def get_nearest_neighbor_idxs(self, normalized_x):

        #expensive multiplication hence device dependent. TODO: use LSH to approximate this
        #(batch_size, key_dim)x(memory_dim, key_dim)
        similarities = None
        with tf.device(self.nn_device):
            #stop gradient to make hash function independent of the nearest neighbor search
            similarities = tf.matmul(normalized_x, self.mem_keys, transpose_b=True, name='nn_mmul')
        _, nearest_neighbor_idxs = tf.nn.top_k(similarities, k=self.choose_k, name='nn_topk')

        neighbors_mem_keys = tf.gather(self.mem_keys, nearest_neighbor_idxs, name='query_mem_keys_val')
        neighbors_similarities = tf.matmul(tf.expand_dims(normalized_x, 1), neighbors_mem_keys, transpose_b=True, name='query_similarities_val')
        neighbors_similarities = tf.squeeze(neighbors_similarities, [1], 'hint_pool_sims_val')

        return neighbors_similarities, nearest_neighbor_idxs

    def get(self):
        return self.mem_keys, self.mem_vals, self.mem_age, self.recent_idx

    def set(self, mem_keys, mem_vals, mem_age, recent_idx):
        return tf.group(self.mem_keys.assign(mem_keys),
                        self.mem_vals.assign(mem_vals),
                        self.mem_age.assign(mem_age),
                        self.recent_idx.assign(recent_idx) if recent_idx is not None else tf.group())

    def clear_memory(self):
        return tf.variables_initializer([self.mem_keys, self.mem_vals, self.mem_age, self.recent_idx])