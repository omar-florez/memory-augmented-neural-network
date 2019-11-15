import logging
import os
import random
import sys
import shutil

#environment setup
sys.path.append('.')
import tensorflow as tf
import numpy as np
import collections
from operator import itemgetter

from utils import data_utils
from models import model_mem as mm

from dataset.SyntheticDataset import SyntheticData
from tensorflow.contrib.tensorboard.plugins import projector
from collections import Counter
import ipdb

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('key_dim', 32, 'dimension of each hash key')
#we will create datasets of K=episode_length/episode_width number of elements per class. We should have at least that
#number of elements to pick from the synthetic dataset
tf.flags.DEFINE_integer('episode_length', 100, 'total number of observations in an episode (considering all minibatches)')
tf.flags.DEFINE_integer('episode_width', 5, 'number of unique classes within an episode')
tf.flags.DEFINE_integer('memory_size', 20, 'memories size')
tf.flags.DEFINE_integer('batch_size', 4, 'batch size')
tf.flags.DEFINE_integer('number_episodes', 1000, 'number of episodes')
tf.flags.DEFINE_integer('validation_frequency', 5, 'validation frequency')
tf.flags.DEFINE_integer('save_frequency', 10, 'Frequency to save model checkpoints')
tf.flags.DEFINE_integer('summary_frequency', 10, 'number of iterations to save summaries')
tf.flags.DEFINE_integer('validation_length', 10, 'number of episodes used to compute validation accuracy')
tf.flags.DEFINE_integer('seed', 888, 'randon seed generator')
tf.flags.DEFINE_string('save_dir', './temp/checkpoints_synthetic', 'directory to save the models')
tf.flags.DEFINE_string('summary_dir', './temp/summaries', 'directory to save the models')
tf.flags.DEFINE_bool('restore', False, 'Restore entire model')
tf.flags.DEFINE_bool(''
                     '', False, 'use locality sensitive hashing')
tf.flags.DEFINE_bool("keys_are_trainable", False, 'set to true to allow backprop to reach the keys and thus update them')
tf.flags.DEFINE_bool('clear_memory_train', False, 'clear memories before running a training step')
tf.flags.DEFINE_bool('clear_memory_validation', False, 'clear memories before running a validation step')
tf.flags.DEFINE_string('data_dir', None, 'to set a data directory')
tf.flags.DEFINE_string('model_name', 'synthetic_dataset', 'name of the model, used to save checkpoints')
tf.flags.DEFINE_integer('choose_k', 256, 'size of the neighborhood to look for content in memories')

def hist_str(array):
    output = sorted(Counter(np.array(array).ravel()).items(), key=itemgetter(0))
    return str(output)

class Trainer(object):
    def __init__(self, train_data, valid_data, data_info, input_dim, output_dim=None):
        self.train_data = train_data
        self.valid_data = valid_data
        self.input_dim = input_dim
        self.data_info = data_info

        self.key_dim = FLAGS.key_dim                                                                                #128
        self.episode_length = FLAGS.episode_length                                                                  #30
        self.episode_width = FLAGS.episode_width                                                                    #5
        self.batch_size = FLAGS.batch_size                                                                          #16
        self.memory_size = self.episode_length * self.batch_size if FLAGS.memory_size is None else FLAGS.memory_size#30*16=8192
        self.output_dim = FLAGS.episode_width if output_dim is None else output_dim
        self.use_lsh = FLAGS.use_lsh
        self.create_folders()

    def create_folders(self):
        if os.path.exists(FLAGS.save_dir):
            shutil.rmtree(FLAGS.save_dir, ignore_errors=True)

        folders_to_create = [FLAGS.save_dir]
        for folder in folders_to_create:
            if not os.path.exists(folder):
                os.makedirs(folder)

    # driver method to train models
    def run(self):
        train_data, valid_data = self.train_data, self.valid_data
        input_dim, output_dim = self.input_dim, self.output_dim
        key_dim, episode_length = self.key_dim, self.episode_length
        episode_width, memory_size = self.episode_width, self.memory_size
        batch_size = self.batch_size
        train_size = len(train_data)
        val_size = len(valid_data)
        output_dim = episode_width
        dataset_intersection = set(train_data.keys()).intersection(set(valid_data.keys()))


        logging.info('train size (number of labels): %d' %train_size)
        logging.info('valid size (number of labels): %d' %val_size)
        logging.info('input_dim: %d' %input_dim)
        logging.info('output_dim: %d' %output_dim)
        logging.info('memory_size: %d' %memory_size)
        logging.info('batch_size: %d' %batch_size)
        logging.info('episode length: %d' %episode_length)
        logging.info('episode_width: %d' %episode_width)
        logging.info('choose_k: %d' %FLAGS.choose_k)

        self.model = self.get_model()
        self.model.setup()

        sess = tf.Session()

        sess.run(tf.global_variables_initializer())

        #summary:
        train_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/train', sess.graph)
        #test_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/test', sess.graph)
        #write/restore model
        saver = tf.train.Saver(max_to_keep=10)
        ckpt = None

        if FLAGS.save_dir:
            ckpt = tf.train.get_checkpoint_state(FLAGS.save_dir)
        if ckpt and ckpt.model_checkpoint_path and FLAGS.restore:
            logging.info('restoring model from: %s' %ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)

        logging.info('start training...')

        losses=[]
        for i in range(FLAGS.number_episodes):
            #len(x) = len(y) = episode_length
            #len(x[0]) = len(y[0]) = batch_size observations
            #len(np.unique([yy[0] for yy in y])) = episode_width

            seqlen = None
            #-----------------------------------------------------------------------------------------------------------
            # Training step:
            #       train_data.keys(): N classes
            x, y = self.sample_episode_batch(train_data, episode_length, episode_width, batch_size)

            #print('Counting label distribution in training labels for this episode: ', Counter(np.array(y).ravel()))
            loss, y_train, summary = self.model.step_training(sess, x, y, seqlen, clear_memory=FLAGS.clear_memory_train)
            curr_memory = sess.run([self.model.mem_keys, self.model.mem_vals, self.model.mem_age])
            #losses.append(loss)

            #-----------------------------------------------------------------------------------------------------------
            # Validation step:
            #       collect k-shot results during the FLAGS.validation_length experiments (episodes)
            correct_by_shot = dict((k, []) for k in range(self.episode_width + 1))
            correct = []
            embedding_keys = []
            embedding_labels = []
            #if True:
            if i%FLAGS.validation_frequency==0 and i>0:
                losses=[]
                #logging.info('validation batch %d, average train loss %f' %(i, np.mean(loss)))
                for _ in range(FLAGS.validation_length):
                    #keep track of counts or shots for labels
                    count_in_this_episode = [dict() for _ in range(batch_size)]

                    #batch size : 1
                    #len(x)     : 30 minibatches of batch_size each one
                    #x[0]       : (1, 2080)
                    #x1, y1 = self.sample_episode_batch(valid_data, episode_length, episode_width, 1)
                    #pp [(i, np.mean(np.vstack(x1)[np.vstack(y1).ravel() == i], axis=0)) for i in np.unique(y1)]

                    x, y = self.sample_episode_batch(valid_data, episode_length, episode_width, 1)

                    #print('Counting validation labels: ', Counter(np.array(y).ravel()))
                    y_val = self.model.step_validation2(sess, x, y,
                                                        clear_memory=FLAGS.clear_memory_validation)
                    #y_val, _, _ = self.model.step_testing(sess, x)
                    prediction = np.mean(np.equal(np.array(y_val), y))
                    logging.info('[VALIDATION] prediction:\t' + str(prediction))
                    correct.append(prediction)

                    # Compute embedding for later visualization
                    #(30, 128)
                    val_keys = sess.run(self.model.memory.fetched_keys,
                                        feed_dict = {self.model.x: np.array(x)[:, 0, :32],
                                                     self.model.y: np.array(y).ravel()})

                    embedding_keys.append(val_keys)
                    embedding_labels.append(y)
                    #---------------------------------------------------------------------------------------------------
                    # K-shot inference:
                    for yy, yy_pred in zip(y, y_val):
                        for k, (yyy, yyy_pred) in enumerate(zip(yy, yy_pred)):
                            yyy, yyy_pred = int(yyy), int(yyy_pred)
                            count_shot = count_in_this_episode[k].get(yyy, 0)
                            if count_shot in correct_by_shot: #just in case
                                correct_by_shot[count_shot].append(yyy == yyy_pred)
                            count_in_this_episode[k][yyy] = count_shot + 1
                #-------------------------------------------------------------------------------------------------------
                # reporting:
                #ipdb.set_trace()
                logging.info('overall validation accuracy: %f' %np.mean(correct))
                logging.info('%d-shot: %.3f | '*(self.episode_width+1),
                             *sum([[k, np.mean(correct_by_shot[k])] for k in range(self.episode_width+1)], []))
                #ipdb.set_trace()
                # embedding_keys = np.concatenate(embedding_keys)
                # embedding_labels = np.concatenate(embedding_labels)
                # np.save('./temp/summaries/embedding_keys_%d.npy' %i, embedding_keys)
                # np.save('./temp/summaries/embedding_labels_%d.npy' %i, embedding_labels)


            if i % (FLAGS.save_frequency) == 0 and i > 0:
                if saver and FLAGS.save_dir:
                    saved_file = saver.save(sess, os.path.join(FLAGS.save_dir, 'model.ckpt'),
                                            global_step = self.model.global_step)
                    logging.info('savel model to %s' %saved_file)
                #ipdb.set_trace()


    def get_model(self):
        '''
        Perform training given episodes
            x = (episode_length, batch_size, image_dim)
            y = (episde_length, batch_size)
        :return:
        '''
        #number of different keys in the hash table given training episodes
        self.vocab_size = self.episode_width * self.batch_size
        model = mm.Model(self.input_dim,
                         self.output_dim,
                         self.key_dim,
                         self.memory_size,
                         self.vocab_size,
                         use_lsh = self.use_lsh,
                         embedder_algo = 'mlp',
                         keys_are_trainable = FLAGS.keys_are_trainable,
                         choose_k=FLAGS.choose_k,
                         seq_max_len=2)
        return model

    def sample_episode_batch(self, data, episode_length, episode_width, batch_size):
        #np.unique([yy[0] for yy in y]):
        #       [1027, 2132, 3115, 3281, 3784]
        # np.sort(bb)
        #       [496, 496, 496, 496, 496, 496, 1383, 1383, 1383, 1383, 1383,
        #        1383, 2072, 2072, 2072, 2072, 2072, 2072, 2125, 2125, 2125, 2125,
        #        2125, 2125, 2563, 2563, 2563, 2563, 2563, 2563]
        #   episode_width*

        """
        Builds an episodic dataset. An episode is a collection of episode_length batches. Each batch contains
        batch_size number of observations and labels such that the distinct number of labels across batches is
        equal to episode_length

        Generates a random batch for training or validation
        Structures each element of the batch as an 'episode'
        Each episode contains episode_length examples and
        episode_width distinct labels.
        Args:
          data: A dictionary mapping <label_idx, matrix of observations>
          episode_length: Number of minibatches forming an episode
          episode_width: Distinct number of labels in each episode
          batch_size: Number of mini-batches
        Returns:
          A tuple (x, y) where x is a list of batches of examples
          with size episode_length and y is a list of batches of labels.
        """

        episodes_x = [[] for _ in range(episode_length)]
        episodes_y = [[] for _ in range(episode_length)]
        assert len(data) >= episode_width

        #---------------------------------------------------------------------------------------------------------------
        # Choose keys randomnly
        keys = [key for key in data.keys() if len(data[key])>episode_width]
        episode_labels = random.sample(keys, episode_width)
        support_per_all_labels = {lab:len(data[lab]) for lab in data.keys()}
        support_per_episode_labels = {lab:len(data[lab]) for lab in episode_labels}

        #form batch_size minibatches to form an episode
        for b in range(batch_size):
            remainder = episode_length % episode_width
            remainders = [0] * (episode_width - remainder) + [1] * remainder

            #random.sample(population, k): Return a k length list of unique elements chosen from the population sequence.
            #   Used for random sampling without replacement.
            #   Sample this amount of times: (episode_length - remainder) / episode_width)
            #   but add 1 elements when remainders is different than zero

            episode_x = []
            for lab, r in zip(episode_labels, remainders):
                #K: number of elements per class = episode_length/episode_width
                K = int(r + (episode_length - remainder) / episode_width)
                #skip if there are not many observations (at leat K observations) in class 'lab'
                if K > len(list(data[lab])):
                    continue
                #print('len(list(data[lab])): ', len(list(data[lab])), ' K: ', K)
                samples = random.sample(list(data[lab]), K)
                episode_x.append(samples)

            # (x, i, ii): <observation, character local index, observation local index>
            episode = sum([[(x, class_id, ii) for ii, x in enumerate(xx)] for class_id, xx in enumerate(episode_x)], [])
            random.shuffle(episode)
            # Arrange episode so that each distinct label is seen before moving to
            # 2nd showing
            episode.sort(key=lambda elem: elem[2])

            # if len(episode) == 0:
            #     ipdb.set_trace()
            assert len(episode) == episode_length

            # iterate over episode_length=30 episodes this way we will uniquely see a label 'episode_width'
            # number of times across different 'episode_width' number of batches
            for i in range(episode_length):
                episodes_x[i].append(episode[i][0])
                ##episodes_y[i].append(episode[i][1] + b * episode_width)     #???
                episodes_y[i].append(episode_labels[episode[i][1]])  # ???

        batch_x = [np.array(xx).astype('float32') for xx in episodes_x]
        batch_y = [np.array(yy).astype('int32') for yy in episodes_y]

        #batch_x: 30 x (16, 2080)
        #batch_y: 30 x (16,)
        return (batch_x, batch_y)




def visualize_embeddings_in_tensorboard(final_embedding_matrix, metadata_path, dir_path):
    """
    view the tensors in tensorboard with PCA/TSNE
    final_embedding_matrix: embedding vector
    metadata_path: path to the vocabulary indexing the final_embedding_matrix
    """
    with tf.Session() as sess:
        embedding_var = tf.Variable(final_embedding_matrix, name='embedding')
        sess.run(embedding_var.initializer)

        config = projector.ProjectorConfig()

        # add embedding to the config file
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name

        # link this tensor to its metadata file, in this case the first 500 words of vocab
        embedding.metadata_path = metadata_path

        visual_summary_writer = tf.summary.FileWriter(dir_path)

        # saves a configuration file that TensorBoard will read during startup.
        projector.visualize_embeddings(visual_summary_writer, config)

        saver_embed = tf.train.Saver([embedding_var])
        saver_embed.save(sess, dir_path + '/visual_embed.ckpt', 1)

        visual_summary_writer.close()

# Train:
#python run/train_labels.py --memory_size=7000 --batch_size=16 --validation_length=50 --episode_width=5 --episode_length=30

#python run/train_synthetic.py --batch_size=20 --memory_size=100 --clear_memory_validation=False --choose_k=2

#python run/train_synthetic.py --batch_size=20 --memory_size=100 --clear_memory_validation=False --clear_memory_train=True --choose_k=50 --episode_width=10 --validation_length=1
#tensorboard --logdir=./temp/summaries/train/
if __name__ == '__main__':
    #np.set_printoptions(threshold=np.NaN)
    #np.set_printoptions(linewidth=np.NaN)

    num_classes     = 10   # linear sequence or not
    input_dim       = 2

    # Parameters
    learning_rate   = 0.01
    training_steps  = 100000
    batch_size      = 128
    display_step    = 200
    testing_step    = 500

    dataset = SyntheticData(num_classes=num_classes, num_elements_per_class=200, std=0.1, seed=0)
    #test and validation dataset
    dataset_info = dataset.get_train_test_val_data()
    train_data = dataset_info[0]
    train_label = dataset_info[1]
    train_seqlen = dataset_info[2]
    test_data = dataset_info[3]
    test_label = dataset_info[4]
    test_seqlen = dataset_info[5]
    val_data = dataset_info[6]
    val_label = dataset_info[7]
    val_seqlen = dataset_info[8]
    vocab_size = dataset_info[9]
    seq_max_len = dataset_info[10]

    len_intersection_train_test = len(set(train_label.ravel()).intersection(set(test_label.ravel())))
    len_intersection_train_val = len(set(train_label.ravel()).intersection(set(val_label.ravel())))
    print('Size of class intersection between train and test: %d' %len_intersection_train_test)
    print('Size of class intersection between train and validation: %d' % len_intersection_train_val)
    #-------------------------------------------------------------------------------------------------------------------
    # Create dictionary dataset
    # train_data:                       a dictionary of N tokens labels
    #   x_train_intent.keys()           [0, 1, 2, 3, 4, 5, n_classes]
    #   len(x_train_intent[k]):         [1868, 398, 151, 717, 41, 184, 266, ...
    #   x_train_intent[0][0].shape      an array of size (seq_len x max_word_length)
    x_train = {}
    for class_id in np.unique(train_label):
        idx_class = train_label == class_id
        x_train[class_id] = train_data[idx_class.ravel()]
    x_test = {}
    for class_id in np.unique(test_label):
        idx_class = test_label == class_id
        x_test[class_id] = test_data[idx_class.ravel()]
    x_val = {}
    for class_id in np.unique(val_label):
        idx_class = val_label == class_id
        x_val[class_id] = val_data[idx_class.ravel()]

    #-------------------------------------------------------------------------------------------------------------------
    # Run
    logging.basicConfig(level=logging.INFO, filename='./myapp.log', filemode='w')
    #logging.basicConfig(level=logging.INFO)
    trainer = Trainer(x_train, x_val, None, input_dim=input_dim, output_dim=num_classes)
    trainer.run()


