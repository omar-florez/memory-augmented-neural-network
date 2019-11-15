import tensorflow as tf
import os
import ipdb
import sys
import numpy as np
import random

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('input_dim', 2, 'dimension of input data')
tf.flags.DEFINE_integer('key_dim', 128, 'dimension of keys used in map memories')
tf.flags.DEFINE_integer('episode_length', 128, 'length of episode')
tf.flags.DEFINE_integer('episode_width', 5, 'number of distinct classes in episode')
tf.flags.DEFINE_integer('memory_size', 7000, 'number of slots in memories')
tf.flags.DEFINE_integer('batch_size', 16, 'batch size')
tf.flags.DEFINE_integer('number_episodes', 100000, 'number of training episodes')
tf.flags.DEFINE_integer('validation_frequency', 10, 'compute validation accuracy every number of training episodes')
tf.flags.DEFINE_integer('save_frequency', 100, 'number of iterations to save a model')
tf.flags.DEFINE_integer('validation_length', 10, 'number of episodes used to compute validation accuracy')
tf.flags.DEFINE_integer('seed', 888, 'randon seed generator')
tf.flags.DEFINE_integer('choose_k', 256, 'neighborhood size')
tf.flags.DEFINE_integer('output_dim', 10, 'number of classes')
tf.flags.DEFINE_string('save_dir', '../files/trained_models', 'directory to save the models')
tf.flags.DEFINE_string('saved_model', 'model.ckpt-822030', 'name of ckpt model')
##tf.flags.DEFINE_string('embedder_algo', 'lstm', 'name of data encoder')
tf.flags.DEFINE_bool('use_lsh', False, 'use locality sensitive hashing')
tf.flags.DEFINE_string('data_dir', './data/labels_characters', 'folder containing tokens datasets')
tf.flags.DEFINE_string('dataset_name', 'tokens', 'name of the dataset')
tf.flags.DEFINE_bool('keys_are_trainable', False, 'to update hash keys as part of the training')
tf.flags.DEFINE_integer('seq_max_len', 32, 'maximun length of input sequence')
tf.flags.DEFINE_integer('num_tables', 5, 'Number of hash tables for LSH')
tf.flags.DEFINE_string('model_name', 'lsh', 'name of current model')
tf.flags.DEFINE_string('info', '', 'Information about current run')
tf.flags.DEFINE_bool('clear_memory', False, 'clear current memories state?')
tf.flags.DEFINE_bool('train_traditionally', False, 'train traditionally')
tf.flags.DEFINE_integer('num_epochs', 2000, 'Number of epochs')
tf.flags.DEFINE_string('summary', '%s_%s' % ('generic', str(random.randint(1,100000))), 'summary name for tensorboard visualization')
tf.flags.DEFINE_integer('k_fold', 1, 'test iterations')
tf.flags.DEFINE_integer('classes', 10, 'total number of classes')
tf.flags.DEFINE_string('saved_folder', "./saved/figures/experiments", "where to save summary file")
tf.flags.DEFINE_string('experiment_name', "memory_size", "name of summary file")
tf.flags.DEFINE_bool('run_plot_boundary', False, "boundary of plot file")
tf.flags.DEFINE_string('model_id', '', "Name to use in tensorboard")

# dataset setting:
tf.flags.DEFINE_integer('seq_len_chars', 32, 'total number of classes')
tf.flags.DEFINE_integer('seq_len_words', 10, 'total number of classes')


# encoder settings:
tf.flags.DEFINE_integer('num_layers', 1, "Number of LSTM layers used in the encoder")
tf.flags.DEFINE_bool('bidirectional_rnn', False, "Use bidirectional RNN units")
tf.flags.DEFINE_string('input_encoding', 'word_random', "Type of word encoding used")

tf.flags.DEFINE_bool('write_memory_stats', False, 'write testing dataset predicted by current memories state')
tf.flags.DEFINE_bool('run_local', False, 'to set some paths to run in local computer and save summary')

# Server
#tf.flags.DEFINE_string('tensorboard_summary_folder', '/home/ubuntu/log_dir/summary/', 'Name to use in tensorboard')
# Local
tf.flags.DEFINE_string('tensorboard_summary_folder', './saved/summary/', 'Name to use in tensorboard')


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models import model_lstm
from models import model_mem
from models import model_mlp

import random

class Loader(object):
    def __init__(self):
        tf.reset_default_graph()

        #model params:
        #then goal is to have at least self.episode_length/self.episode_width observations per class
        self.episode_length = 30
        self.episode_width = 10

        self.batch_size = 16
        #self.valid_dataset = self.load_validation_dataset()

    def get_model(self, model_name):
        tf.reset_default_graph()
        if model_name == 'mlp':
            return model_mlp.MLP(FLAGS.input_dim,
                                 FLAGS.output_dim,
                                 learning_rate=0.01)
        if model_name == 'lth':
            model = model_mem.Model(FLAGS.input_dim,
                                    FLAGS.output_dim,
                                    FLAGS.key_dim,
                                    FLAGS.memory_size,
                                    vocab_size,
                                    use_lsh=FLAGS.use_lsh,
                                    embedder_algo=FLAGS.embedder_algo,
                                    keys_are_trainable=FLAGS.keys_are_trainable,
                                    choose_k=FLAGS.choose_k,
                                    seq_max_len=FLAGS.seq_max_len)
            return model
        if model_name == 'baseline':
            vocab_size = FLAGS.episode_width * FLAGS.batch_size
            model = mm_baseline.Model(FLAGS.input_dim,
                                      FLAGS.output_dim,
                                      FLAGS.key_dim,
                                      vocab_size,
                                      embedder_algo=FLAGS.embedder_algo,
                                      seq_max_len=int(args.seq_max_len))
            return model
        return None


    #-------------------------------------------------------------------------------------------------------------------
    # Model
    def initialize_model(self, model_name='mlp', model=None):
        # get a model object with the model_name
        if model == None and model_name != None:
            model = self.get_model(model_name)
        #if not name or model is given
        if model == None and model_name == None:
            return None

        self.model = model
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        print('Models trainable params: ', tf.trainable_variables())
        return self.model

    def train(self, x, x_words, y, clear_memory=False, keep_prob=1.0):
        loss, y_preds = self.model.step_training(self.sess, x, x_words, y, keep_prob=keep_prob)
        return y_preds, loss

    # def query(self, x, y=None, debug=False):
    #     y_preds = self.model.step_inference(self.sess, x, y)
    #     ##y_preds2 = self.model.step_inference_neighborsidx(self.sess, x, clear_memory=False, debug=debug)
    #     #ipdb.set_trace()
    #     return y_preds

    def query_val(self, x, x_words, y, debug=False):
        y_preds = self.model.step_validation(self.sess, x, x_words, y)
        return y_preds

    def query_val_info(self, x, x_words, y, debug=False):
        y_preds = self.model.step_validation_info(self.sess, x, x_words, y)
        return y_preds

    def save_model(self, folder_path, counter=None):
        os.makedirs(os.path.dirname(folder_path), exist_ok=True)
        saver = tf.train.Saver()
        if counter:
            saver.save(self.sess, folder_path, global_step=counter)
        else:
            saver.save(self.sess, folder_path)

    # -------------------------------------------------------------------------------------------------------------------
    def load_model(self, save_dir=None, saved_model=None, embedder_algo='lstm'):
        FLAGS.key_dim   = 16
        FLAGS.memory_size = 7000
        FLAGS.choose_k  = 5
        FLAGS.input_dim = 2

        if not save_dir:
            save_dir = FLAGS.save_dir
        if not save_dir:
            saved_model = FLAGS.saved_model

        #FLAGS.input_dim = data_utils.IMAGE_NEW_SIZE**2
        vocab_size      = FLAGS.episode_width * FLAGS.batch_size
        output_dim      = FLAGS.episode_width
        self.x = print('RUN()')

        #define model:
        vocab_size = FLAGS.episode_width * FLAGS.batch_size
        self.model = model_mem.Model(FLAGS.input_dim,
                                     output_dim,
                                     FLAGS.key_dim,
                                     FLAGS.memory_size,
                                     vocab_size,
                                     use_lsh=FLAGS.use_lsh,
                                     embedder_algo=embedder_algo,
                                     choose_k=5)
        self.model.setup()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        saver.restore(self.sess, os.path.join(save_dir, saved_model))
        print('List of existing values: ', self.sess.run([self.model.mem_vals]))
        return self.model

    def get(self):
        return self.model

    def inference(self):
        print('---> inference')
        x, y = self.sample_episode_batch(self.valid_data, self.episode_length, self.episode_width, batch_size)
        y_val = self.model.step_validation(self.sess, x, y, clear_memory=True)
        return y_val

    def clear_memory(self):
        self.model.clear_memory(self.sess)

    def get_memory_state(self):
        return self.model.get_memory_state(self.sess)

    # -------------------------------------------------------------------------------------------------------------------
    def run_key_analysis(self, dataset, file_path):
        #keys, values, age = self.get_memory_state()
        x, x_words, y = dataset.x_val, dataset.x_val_words, dataset.y_val
        idx2intent = dataset.idl.idx2intent

        # per-observation assignment
        ypreds, key_idxs, keys, age, sims = self.model.step_key_stats(self.sess, x, x_words)

        key_stats = {}
        #key_stats['avg_precision'] = avg_precision
        for i in range(len(ypreds)):
            key_idx = key_idxs[i]
            if key_idx not in key_stats:
                key_stats[key_idx] = {}
                key_stats[key_idx]['predicted_labels_ids'] = []
                key_stats[key_idx]['actual_labels_ids'] = []
                key_stats[key_idx]['predicted_labels'] = []
                key_stats[key_idx]['actual_labels']    = []
                key_stats[key_idx]['ages'] = []
                key_stats[key_idx]['sims'] = []
                key_stats[key_idx]['utterances'] = []

            key_stats[key_idx]['predicted_labels_ids'].append(ypreds[i])
            key_stats[key_idx]['actual_labels_ids'].append(y[i])

            key_stats[key_idx]['predicted_labels'].append(idx2intent[ypreds[i]])
            key_stats[key_idx]['actual_labels'].append(idx2intent[y[i]])
            key_stats[key_idx]['ages'].append(age[i])
            key_stats[key_idx]['sims'].append(sims[i])

            utterance = dataset.matrix_to_text([x[i]], char_mode=True)[0]
            key_stats[key_idx]['utterances'].append(utterance)

        self.write_html(file_path, key_stats)
        return key_stats

    def write_html(self, file_path, key_stats, folder_name=None):
        html = "<html>@body</html>"
        body = ""

        for key in key_stats:
            agreement = np.equal(key_stats[key]['actual_labels_ids'], key_stats[key]['predicted_labels_ids']).mean()
            body += '<b>Memory Key: {}</b> | Agreement: {:.2f}% <br>'.format(key, agreement*100.0)
            num_neighbors = len(key_stats[key]['utterances'])
            body += '<ul>'+ '\n'
            body += '<b><li> {} | <font color="green">{}</font> | <font color="blue">{}</font></li>{}</b>'.format('Utterance', 'Predicted label', 'Actual label', '\n')
            for i in range(num_neighbors):
                utterance = key_stats[key]['utterances'][i]
                predicted_label = key_stats[key]['predicted_labels'][i]
                actual_label = key_stats[key]['actual_labels'][i]
                miss = '(-)' if predicted_label != actual_label else ''
                body += '<li>{} | <font color="green">{}</font> | <font color="blue">{} {}</font></li>{}'.format(utterance, predicted_label, actual_label, miss, '\n')
            body += '</ul>'+ '\n'
            #body += 'predicted_labels: ' + str(key_stats[key]['predicted_labels']) + '<br>'
            #body += 'actual_labels: ' + str(key_stats[key]['actual_labels']) + '<br>'
            #body += 'ages: ' + str(key_stats[key]['ages']) + '<br>'
            #body += 'sims: ' + str(key_stats[key]['sims']) + '<br>'
            body += '<hr>' + '\n'
        html = html.replace('@body', body)
        with open(file_path, 'w') as file:
            file.write(html)
        return

#   python run/main.py --model_name 'lth' --input_dim 32 --output_dim 133 --batch_size 128 --seq_max_len 32 --embedder_algo 'lstm' --memory_size 10024 --key_dim 256 --choose_k 256 --clear_memory False
if __name__ == '__main__':
    batch_size = 1
    loader = Loader()
    model = loader.load_model()
    loader.random_inference()


