'''
Model definition
@author: Omar U. Florez
'''

#import ipdb
import tensorflow as tf
from models import embedder
from models import embedder_lstm
from models import lstm_encoder
from models import embedder_mlp

from memories import memory

import numpy as np
from collections import Counter
from operator import itemgetter
import logging

import datetime
import ipdb

#logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.INFO, filename='./myapp.log')


class Model(object):
    def __init__(self, data_args, encoder_args, learning_rate=1e-4, model_id='lstm_00.00.00',
                 model_name='lstm', summary_dir=None):

        self.seq_len_chars      = data_args['seq_len_chars']
        self.seq_len_words      = data_args['seq_len_words']
        self.output_dim         = data_args['output_dim']
        self.vocab_size_chars   = data_args['vocab_size_chars']
        self.vocab_size_words   = data_args['vocab_size_words']

        self.key_dim            = encoder_args['mem_size_encoder']
        self.num_layers         = encoder_args['num_layers']
        self.input_encoding     = encoder_args['input_encoding']
        self.encoder            = self.get_encoder(data_args, encoder_args)

        self.learning_rate      = learning_rate

        self.model_name         = model_name
        self.model_id           = model_id

        self.global_step = tf.contrib.framework.get_or_create_global_step()
        self.increment_global_step = self.global_step.assign_add(1)

        self.summary_dir = summary_dir if summary_dir[-1] == '/' else summary_dir + '/'
        self.summary_writer = tf.summary.FileWriter(self.summary_dir + self.model_id)

        self.setup()

    def get_encoder(self, data_args, encoder_args):
        return lstm_encoder.dynamicRNN(data_args,
                                       encoder_args,
                                       run_attention=False)

    #-------------------------------------------------------------------------------------------------------------------
    #Driver method
    def setup(self):
        self.x                  = tf.placeholder(tf.float32, [None, self.seq_len_chars], name='x')
        self.y                  = tf.placeholder(tf.int32, [None], name='y')
        self.keep_prob          = tf.placeholder_with_default(1.0, shape=(), name='keep_prob')
        self.x_words            = tf.placeholder(tf.float32, [None, self.seq_len_words], name='x_words')

        self.loss, self.y_pred  = self.encoder.setup(self.x, self.x_words,
                                                     self.y, keep_prob=self.keep_prob,
                                                     input_encoding=self.input_encoding)
        self.gradient_ops       = self.training_ops(self.loss)

    def training_ops(self, loss):
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        params = tf.trainable_variables()

        gradients = tf.gradients(loss, params)
        gradients, _ = tf.clip_by_global_norm(gradients, 10.0)
        self.grad_summ = {'%s-%s'%(p.name, str(g.shape)):  g for p, g in zip(params, gradients) if g is not None and isinstance(g, tf.Tensor)}

        return opt.apply_gradients(zip(gradients, params), global_step = self.global_step)

    def step_training(self, sess, x, x_words, y, keep_prob=1.0):
        '''
        For baseline model:

        crunching numbers over batches: put training complexity here
        very batch is: (batch_size ,input_dim)
        :param sess: 
        :param x: A list of batches of observations defining the episode
        :param y: A list of batches of labels corresponding to x
        :return: 
        '''
        episode_count = sess.run(self.global_step)
        ops = [self.loss,
               self.gradient_ops,
               self.y_pred,
               self.grad_summ]
        
        losses, y_preds = [], []
        for xx, xx_words, yy in zip(x, x_words, y):
            outputs = sess.run(ops, feed_dict={self.x: xx[:, :self.seq_len_chars],
                                               self.x_words: xx_words[:, :self.seq_len_words],
                                               self.y: yy,
                                               self.encoder.seqlen: np.zeros(len(xx)),
                                               self.keep_prob: keep_prob})

            loss_train = outputs[0]
            y_pred = outputs[2]
            grad_summ = outputs[3]
            #print("Loss: {}, xx_size: {}, xx_words_size: {}".format(loss_train, len(xx) ,len(xx_words)))

            losses.append(loss_train)
            y_preds.append(y_pred)

            precision = np.mean(yy == y_pred)

            summary = tf.Summary()
            #summary.value.add(tag='Loss/avg_gradient', simple_value)
            summary.value.add(tag='Precision/precision_training', simple_value=precision)
            summary.value.add(tag='Loss/loss_train', simple_value=loss_train)
            self.summary_writer.add_summary(summary, episode_count)

            #--ValueError: autodetected range of [nan, nan] is not finite
            # for grad_name in grad_summ:
            #     self.summary_histogram(self.summary_writer,
            #                            "GradientsEncoder/%s"%grad_name,
            #                            np.ravel(grad_summ[grad_name]),
            #                            episode_count)
            self.summary_writer.flush()
            sess.run(self.increment_global_step)
            episode_count += 1
        return losses, y_preds

    def step_validation(self, sess, x, x_words, y, keep_prob=1.0):

        ops = [self.y_pred,
               self.encoder.probabilities]
        y_preds = []
        hits = []
        summary = tf.Summary()

        for xx, xx_words, yy in zip(x, x_words, y):
            # if array is 1D, convert it to 2D
            xx = xx.reshape([-1, len(xx)]) if xx.ndim == 1 else xx
            xx_words = xx_words.reshape([-1, len(xx_words)]) if xx_words.ndim == 1 else xx_words
            yy = [yy] if yy.ndim == 0 else yy

            outputs = sess.run(ops, feed_dict={self.x: xx[:, :self.seq_len_chars],
                                               self.x_words: xx_words[:, :self.seq_len_words],
                                               self.y: yy,
                                               self.keep_prob: keep_prob,
                                               self.encoder.seqlen: np.zeros(len(xx))})
            y_pred = outputs[0]
            y_probs = outputs[1]
            y_preds.append(y_pred)
            hits.append(yy==y_pred)

        precision = np.mean(hits)
        episode_count = sess.run(self.global_step)
        summary = tf.Summary()
        summary.value.add(tag='Precision/precision_validation', simple_value=precision)
        self.summary_writer.add_summary(summary, episode_count)
        self.summary_writer.flush()
        return y_preds

    #
    # def step_inference(self, sess, x, y):
    #     '''
    #     Inference without knowing value of label y
    #     :param sess: current session
    #     :param x: input episode (list of mini-batches)
    #     :param clear_memory: to clear memories
    #     :return: list of predicted values
    #     '''
    #
    #     ops = [self.y_val]
    #     y_preds = []
    #     for xx, yy in zip(x, y):
    #         outputs = sess.run(ops, feed_dict={self.x: xx[:, :self.seq_max_len],
    #                                            #self.y: yy,
    #                                            self.encoder.seqlen: np.zeros(len(xx))}) #np.zeros(len(xx))})
    #         y_pred = outputs[0]
    #         y_preds.append(y_pred)
    #
    #     average_precision = np.mean(np.array(y_preds) == np.array(y))
    #
    #     summary = tf.Summary()
    #     summary.value.add(tag='Precision/precision_validation', simple_value=average_precision)
    #     episode_count = sess.run(self.global_step)
    #     self.summary_writer.add_summary(summary, episode_count)
    #     self.summary_writer.flush()
    #     return y_preds

    def clear_memory(self):
        print('LSTM CLEAR MEMORY ')
        return None

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

#-----------------------------------------------------------------------------------------------------------------------
# Console:
color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)
def colorize(string, color, bold=False, highlight=False):
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

# Run:
#   export PYTHONPATH=.
#   python3 models/model_lstm.py --info 'debug: mem_size=64, hidden=64, dropout=0.8, num_layers=3' --run_local
#   python3 models/model_lstm.py --key_dim 256 --input_encoding 'word_glove' --info 'encoding=word_glove key_dim=256, seq_len_words = 10' --run_local

#   export CUDA_VISIBLE_DEVICES=''; python3 models/model_lstm.py --input_encoding 'word_random' --num_layers 1 --keep_prob 0.5 --key_dim 256 --info 'LSTM, mem_size=256, dropout=0.5, num_layers=1, unidirectional_rnn, clip_gradients=10.0, encoding=word_random' --run_local
#   export CUDA_VISIBLE_DEVICES=''; python3 models/model_lstm.py --input_encoding 'word_glove' --num_layers 1 --keep_prob 0.5 --key_dim 256 --info 'LSTM, mem_size=256, dropout=0.5, num_layers=1, unidirectional_rnn, clip_gradients=10.0, encoding=word_glove' --run_local
#   export CUDA_VISIBLE_DEVICES=''; python3 models/model_lstm.py --input_encoding 'character_random' --num_layers 1 --keep_prob 0.5 --key_dim 256 --info 'LSTM, mem_size=256, dropout=0.5, num_layers=1, unidirectional_rnn, clip_gradients=10.0, encoding=character_random' --run_local
#   export CUDA_VISIBLE_DEVICES=''; python3 models/model_lstm.py --input_encoding 'character_onehot' --num_layers 1 --keep_prob 0.5 --key_dim 256 --info 'LSTM, mem_size=256, dropout=0.5, num_layers=1, unidirectional_rnn, clip_gradients=10.0, encoding=character_onehot' --run_local

#   export CUDA_VISIBLE_DEVICES=''; python3 models/model_lstm.py --input_encoding 'word_glove' --num_layers 3 --keep_prob 0.5 --key_dim 256 --info 'LSTM, mem_size=256, dropout=0.5, num_layers=3, unidirectional_rnn, clip_gradients=10.0, encoding=word_glove' --run_local
#   export CUDA_VISIBLE_DEVICES=''; python3 models/model_lstm.py --input_encoding 'word_glove' --num_layers 3 --keep_prob 0.5 --key_dim 256 --bidirectional_rnn --info 'LSTM, mem_size=256, dropout=0.5, num_layers=3, bidirectional_rnn, clip_gradients=10.0, encoding=word_glove' --run_local
#   export CUDA_VISIBLE_DEVICES=''; python3 models/model_lstm.py --input_encoding 'word_random' --num_layers 3 --keep_prob 0.5 --key_dim 256 --info 'LSTM, mem_size=256, dropout=0.5, num_layers=3, unidirectional_rnn, clip_gradients=10.0, encoding=word_random' --run_local
#   export CUDA_VISIBLE_DEVICES=''; python3 models/model_lstm.py --input_encoding 'word_random' --num_layers 3 --keep_prob 0.5 --key_dim 256 --bidirectional_rnn --info 'LSTM, mem_size=256, dropout=0.5, num_layers=3, bidirectional_rnn, clip_gradients=10.0, encoding=word_random' --run_local
#   export CUDA_VISIBLE_DEVICES=''; python3 models/model_lstm.py --input_encoding 'character_random' --num_layers 3 --keep_prob 0.5 --key_dim 256 --info 'LSTM, mem_size=256, dropout=0.5, num_layers=3, unidirectional_rnn, clip_gradients=10.0, encoding=character_random' --run_local
#   export CUDA_VISIBLE_DEVICES=''; python3 models/model_lstm.py --input_encoding 'character_random' --num_layers 3 --keep_prob 0.5 --key_dim 256 --bidirectional_rnn --info 'LSTM, mem_size=256, dropout=0.5, num_layers=3, bidirectional_rnn, clip_gradients=10.0, encoding=character_random' --run_local
#   export CUDA_VISIBLE_DEVICES=''; python3 models/model_lstm.py --input_encoding 'character_onehot' --num_layers 3 --keep_prob 0.5 --key_dim 256 --info 'LSTM, mem_size=256, dropout=0.5, num_layers=3, unidirectional_rnn, clip_gradients=10.0, encoding=character_onehot' --run_local
#   export CUDA_VISIBLE_DEVICES=''; python3 models/model_lstm.py --input_encoding 'character_onehot' --num_layers 3 --keep_prob 0.5 --key_dim 256 --bidirectional_rnn --info 'LSTM, mem_size=256, dropout=0.5, num_layers=3, bidirectional_rnn, clip_gradients=10.0, encoding=character_onehot' --run_local

#   export CUDA_VISIBLE_DEVICES=''; python3 models/model_lstm.py --input_encoding 'word_random' --num_layers 5 --keep_prob 0.5 --key_dim 64 --bidirectional_rnn --info 'LSTM, mem_size=64, dropout=0.5, num_layers=5, bidirectional_rnn, clip_gradients=10.0, encoding=word_random' --run_local

# cd ~/Documents/OneDrive/workspace/WorkspaceCapitalOne/learning_to_hash_labels/saved
# tensorboard --logdir './summary'
if __name__ == '__main__':
    from dataset.SyntheticDataset import SyntheticData
    from dataset.IntentDataset import labelsequenceData
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len_chars',    default=32,       type=int)
    parser.add_argument('--seq_len_words',    default=10,       type=int)
    parser.add_argument('--output_dim',       default=133,      type=int)
    parser.add_argument('--key_dim',          default=64,       type=int)
    parser.add_argument('--encoder_algo',     default='lstm')
    parser.add_argument('--seq_max_len',      default=32,       type=int)
    parser.add_argument('--batch_size',       default=128,      type=int)
    parser.add_argument('--num_steps',        default=40000,    type=int)
    parser.add_argument('--info',             default='')
    parser.add_argument('--num_layers',       default=1,        type=int)
    parser.add_argument('--keep_prob',        default=0.8,      type=float)

    parser.add_argument('--model_name',       default='lstm')
    parser.add_argument('--model_id',         default='lstm_00.00.00')
    parser.add_argument('--input_encoding',   default='word_glove')     #word_glove, word_random, character_random, character_onehot
    parser.add_argument('--bidirectional_rnn',  action='store_true')

    # server
    parser.add_argument('--tensorboard_summary_folder', default='/home/ubuntu/log_dir/summary')
    parser.add_argument('--data_dir', default='/home/ubuntu/learning_to_hash_labels/data')
    parser.add_argument('--run_local', action='store_true')

    args = parser.parse_args()
    if args.run_local:
        # local
        args.tensorboard_summary_folder = './saved/summary'
        args.data_dir = './data'

    #-------------------------------------------------------------------------------------------------------------------
    # dataset
    dataset = labelsequenceData(data_dir=args.data_dir)
    vocab_size_chars = dataset.get_vocab_size()
    vocab_size_words = dataset.vocab_size_words
    x_val, x_val_words, y_val = dataset.x_val, dataset.x_val_words, dataset.y_val
    data_args = {'vocab_size_chars':    vocab_size_chars,
                 'vocab_size_words':    vocab_size_words,
                 'seq_len_chars':       args.seq_len_chars,
                 'seq_len_words':       args.seq_len_words,
                 'output_dim':          args.output_dim}

    now = datetime.datetime.now()
    args.model_id = '%s_%d_%d.%d%s' % (args.model_name, now.day, now.hour, now.minute, args.info)

    #-------------------------------------------------------------------------------------------------------------------
    # model
    tf.reset_default_graph()
    model = Model(data_args,
                  args.key_dim,
                  encoder_algo=args.encoder_algo,
                  model_name=args.model_name,
                  model_id=args.model_id,
                  summary_dir=args.tensorboard_summary_folder,
                  num_layers=args.num_layers,
                  bidirectional_rnn=args.bidirectional_rnn,
                  input_encoding=args.input_encoding)

    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    #-------------------------------------------------------------------------------------------------------------------
    # train / test
    for step in range(args.num_steps):
        #x, y = dataset.get_batch(x_train, y_train, args.batch_size)
        x_chars, x_words, y = dataset.get_batch_chars_words(batch_size=args.batch_size, mode="training")
        losses, y_preds = model.step_training(sess, [x_chars], [x_words], [y], keep_prob=args.keep_prob)
        if step%500 == 0 and step > 0:
            avg_precision = np.mean(np.equal(np.array(y_preds).ravel(), np.array(y).ravel()))
            print(colorize('[TRAIN()@%s] Loss: %f Avg_precision: %f Step: %d' % (args.model_id, np.mean(losses), avg_precision, step), 'white', bold=False))

        if step%1000 == 0 and step > 0:
            y_preds = model.step_validation(sess, x_val, x_val_words, y_val)
            avg_precision = np.mean(np.equal(np.array(y_preds).ravel(), np.array(y_val).ravel()))
            print(colorize('[VALIDATION()@%s] Avg_precision: %f Step: %d' % (args.model_id, avg_precision, step), 'blue', bold=True))





