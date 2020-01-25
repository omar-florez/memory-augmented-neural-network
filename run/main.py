import argparse
import numpy as np
import ipdb
import random
import os

from mutils import plot_nn_utils
from sklearn.utils import shuffle
from collections import defaultdict
import matplotlib.pyplot as plt

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import models
from models import model_API
from models import model_mem
from models import model_lstm
from models import model_mlp
from models import lstm_encoder

from dataset import Dataset
import datetime

import pdb
import pandas as pd
import json

parser = argparse.ArgumentParser()
parser.add_argument('--model_name',         default='lsh')              #lsh, lsh, lstm, lsh_range
parser.add_argument('--model_id',           default='model_00.00.00')
parser.add_argument('--key_dim',            default=32,     type=int)
parser.add_argument('--mem_size', nargs='+', default=1000,  type=int)
parser.add_argument('--output_dim',         default=10,     type=int)
parser.add_argument('--choose_k',           default=5,     type=int)
parser.add_argument('--k_fold',             default=1,      type=int)
parser.add_argument('--num_steps',         default=80000,  type=int)
parser.add_argument('--batch_size',         default=32,     type=int)
parser.add_argument('--seq_max_len',        default=2,      type=int)
parser.add_argument('--experiment_name',    default='memory_size')
parser.add_argument('--model_folder',       default='./saved/model')
parser.add_argument('--experiment_folder',  default='./saved/experiments')
parser.add_argument('--saved_folder',       default='./saved/figures/experiments')
parser.add_argument('--encoder_algo',      default='lstm', choices=['lsh', 'mlp', 'lstm'])
parser.add_argument('--num_tables',         default=5)
parser.add_argument('--info',               default='')
# action='store_true': True when present, False otherwise
parser.add_argument('--keys_are_trainable', action='store_true')
parser.add_argument('--clear_memory',       action='store_true')
parser.add_argument('--run_local',          action='store_true')
parser.add_argument('--use_lsh',            action='store_true')
# inference analysis
parser.add_argument('--run_plot_boundary',  action='store_true')
parser.add_argument('--run_keys_clustering',action='store_true')
parser.add_argument('--run_key_projection', action='store_true')
# memories:
parser.add_argument('--memory_type',        default='lth_gmm')  #lsh, lth_gmm, lth
# encoder:
parser.add_argument('--num_layers_encoder', default=1,      type=int)
parser.add_argument('--bidirectional_rnn',  action='store_true')
parser.add_argument('--input_encoding',     default='word_random')  # word_glove, word_random, character_random, character_onehot
parser.add_argument('--keep_prob',          default=0.5,    type=float)
parser.add_argument('--input_encoder_size', default=300,    type=int)   #word glove uses 300 for encoding dimension
# input:
parser.add_argument('--dataset',            default="labels",      choices=['dialog_state_smtd', 'dialogue_states', 'sentiments', 'synthetic_2d', 'SmarterNLP_2019-03-13'])
parser.add_argument('--seq_len_chars',      default=32,             type=int)
parser.add_argument('--seq_len_words',      default=10,             type=int)
#server
#parser.add_argument('--tensorboard_summary_folder',       default='/home/ubuntu/log_dir/summary')
#parser.add_argument('--data_dir',           default='/home/ubuntu/learning_to_hash_labels/data')
#local
parser.add_argument('--tensorboard_summary_folder',       default='./saved/summary')
parser.add_argument('--data_dir',           default='./data')
args, unknown = parser.parse_known_args()

if args.run_local:
    args.tensorboard_summary_folder = './saved/summary'
    args.data_dir = './data'

use_lsh = True if args.model_name == 'lsh' else False
if not os.path.exists(args.saved_folder):
    os.makedirs(args.saved_folder, exist_ok=True)

#-----------------------------------------------------------------------------------------------------------------------
# a common class to abstract the complexity of using different models: LSTM, Memory-Augmented Neural Networks, MLP.
# common methods include:
#       - initialize_model()
#       - train()
#       - save_model()
#       - clear_memory()
model_api = model_API.Loader()

#-----------------------------------------------------------------------------------------------------------------------
# Dataset:
dataset = Dataset.get_dataset(args)
vocab_size_chars = dataset.get_vocab_size()
vocab_size_words = dataset.vocab_size_words
x_val, x_val_words, y_val = dataset.x_val, dataset.x_val_words, dataset.y_val
class_distributions = defaultdict(int)

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

#-----------------------------------------------------------------------------------------------------------------------
# Auxiliary methods:
def get_model():
    '''
    Returns instances to different models:
        - models.model_mem: (our encoder with external memories)
        - models.model_lstm: (a LSTM encoder without external memories)
    which use the generic methods available in models.model_API to train and test
    any available model.

    We consider two types of external memories:
        - Continous memories: GMM? (args.memory_type="lth_gmm")
        - Discrete memories: LTH? (args.memory_type="lth")

    Three types of information are needed:
        - data_args: dataset-related attributes
        - encoder_args: encoder-related attributes
        - memory_args: external memories-related attributes
    '''
    tf.reset_default_graph()

    data_args = {'vocab_size_chars': dataset.get_vocab_size(),
                 'vocab_size_words': dataset.get_vocab_words_size(),
                 'seq_len_chars': args.seq_len_chars,
                 'seq_len_words': args.seq_len_words,
                 'output_dim': args.output_dim}

    encoder_args = {'num_layers': args.num_layers_encoder,
                    'mem_size_encoder': args.key_dim,
                    'input_encoder_size': args.input_encoder_size,
                    'encoder_algo': args.encoder_algo,
                    'input_encoding': args.input_encoding,
                    'bidirectional_rnn': args.bidirectional_rnn}

    memory_args = {'mem_size': args.mem_size[0],
                   'key_dim': args.key_dim,
                   'memory_type': args.memory_type,
                   'keys_are_trainable': args.keys_are_trainable,
                   'choose_k': args.choose_k}

    model_API.FLAGS.model_name = args.model_name

    if args.model_name == 'lth':
        model = model_mem.Model(data_args,
                                 encoder_args,
                                 memory_args,
                                 #model_name=args.model_name,
                                 model_id=args.model_id,
                                 summary_dir=args.tensorboard_summary_folder)
        return model
    if args.model_name == 'lstm':
        model = model_lstm.Model(data_args,
                                 encoder_args,
                                 model_name=args.model_name,
                                 model_id=args.model_id,
                                 summary_dir=args.tensorboard_summary_folder)
        return model
    return None

#-----------------------------------------------------------------------------------------------------------------------
# training:
def train_model(info=''):
    loss_array          = []
    precision_array     = []

    for i in range(args.num_steps):
        # get batches from training, validation, or testing datasets accordingly
        x_chars, x_words, y = dataset.get_batch_chars_words(batch_size=args.batch_size, mode="training")
        # model_api abstracts the use of any model for training and inference
        y_preds, losses = model_api.train([x_chars], [x_words], [y], keep_prob=args.keep_prob)

        # run training:
        if i%100 == 0 and i>0:
            avg_precision = np.mean(np.equal(np.array(y_preds).ravel(), np.array(y).ravel()))
            avg_loss = np.mean(losses)
            precision_array.append([i, avg_precision])
            loss_array.append([i, avg_loss])
            print('[TRAINING] avg_loss: %f avg_precision: %f @ step %d' %(avg_loss, avg_precision, i))

        # run validation:
        if i%500 == 0 and i>0:
            report_inference(model_api, i)
            model_api.save_model(os.path.join(args.model_folder, args.model_id))
            if args.clear_memory == True:
                model_api.clear_memory()

        # plot loss and precision train plots:
        if i%500 == 0 and i>0:
            file_path = '%s/%s-%s-train_loss-%s.png' % (args.saved_folder, args.experiment_name, info, args.model_name)
            plot_nn_utils.plot_function(loss_array, save_filepath=file_path)
            file_path = '%s/%s-%s-train_precision-%s.png' % (args.saved_folder, args.experiment_name, info, args.model_name)
            plot_nn_utils.plot_function(precision_array, save_filepath=file_path)
            if args.clear_memory == True:
                model_API.clear_memory()

    print('Training done')

def project_keys(keys, vals, age, num_components=2, num_classes=133, file_path=None):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    from sklearn import (manifold, datasets, decomposition, ensemble,
                         discriminant_analysis, random_projection)

    N = len(keys)
    area = (10 * np.random.rand(N)) ** 2  # 0 to 15 point radii
    area = np.ones(N)*5

    import matplotlib.cm as cm
    colors = cm.gist_ncar(np.linspace(0, 1, num_classes))
    #colors = np.random.rand(N)

    #projection: t-sne
    tsne_model = TSNE(n_components=num_components, verbose=0, random_state=0,
                      angle=.99, init='pca', early_exaggeration=50, n_iter=2000)
    keys_2D = tsne_model.fit_transform(keys)
    plt.scatter(keys_2D[:,0], keys_2D[:,1], s=area, c=colors[vals], alpha=0.5)
    for i in range(N):
        plt.text(keys_2D[i,0], keys_2D[i,1], str(vals[i]), fontsize=8)
    plt.savefig(file_path + '_tsne.png', dpi = 250)
    plt.close()

    #projection: t-sne
    keys_2D = decomposition.TruncatedSVD(n_components=2).fit_transform(keys)
    plt.scatter(keys_2D[:,0], keys_2D[:,1], s=area, c=colors[vals], alpha=0.5)
    for i in range(N):
        plt.text(keys_2D[i,0], keys_2D[i,1], str(vals[i]), fontsize=8)
    plt.savefig(file_path + '_pca.png', dpi = 250)
    plt.close()

def predict_eno_dataset(model_wrapper, iter, output_folder='./'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # read local data
    dataset_name = 'eno_validation' #args.dataset
    dataset = EnoValidationDataset(data_dir=os.path.join(args.data_dir, dataset_name))
    output_info = model_wrapper.query_val_info(dataset.x_val, dataset.x_val_words, dataset.y_val)
    y_preds = output_info['predictions']
    sims = output_info['sims']

    most_similar = [s[0][0] for s in sims]
    avg_precision = np.mean(np.equal(np.array(y_preds).ravel(), np.array(dataset.y_val).ravel()))
    predicted_class = [dataset.idl.idx2intent[s[0]] for s in y_preds]
    actual_class = [dataset.idl.idx2intent[s] for s in dataset.y_val]

    input_df = pd.read_csv(os.path.join(args.data_dir, dataset_name, 'tokens',
                                        'Retrieve_Transaction-Feb2019-iOS - Retrieval_Transactions-Feb2019.tsv'),
                           sep='\t')
    content = list(zip(input_df['unq_utt_raw_txt'],
                       actual_class,
                       input_df['score'],
                       input_df['SS Model Consensus'],
                       predicted_class,
                       most_similar))
    output_df = pd.DataFrame(content,
                             columns = ['unq_utt_raw_txt', 'lbl_desc', 'score', 'SS Model Consensus', 'hash_nn_predictions', 'score'])

    output_filepath = os.path.join(output_folder, "eno_predictions_{}_{}.csv".format(iter, avg_precision))
    output_df.to_csv(output_filepath, sep='\t')

def report_inference(model_wrapper, iter):
    #-------------------------------------------------------------------------------------------------------------------
    # inference
    y_preds = model_wrapper.query_val(x_val, x_val_words, y_val)
    avg_precision = np.equal(np.concatenate(y_preds), y_val).mean()

    #-------------------------------------------------------------------------------------------------------------------
    # plot decision boundary for 2D datasets
    if args.run_plot_boundary:
        inference = model_wrapper.query
        xplot_train = np.array(x_train).ravel().reshape(-1, 2)
        yplot_train = np.array(y_train).ravel()
        xplot_val = np.array(x_val).ravel().reshape(-1, 2)
        yplot_val = np.array(y_val).ravel()
        file_path = '%s/%s-%s-boundaries-%s.png' % (args.saved_folder, args.experiment_name, args.model_name)
        plot_nn_utils.plot_datasets_and_boundary(xplot_train, yplot_train, xplot_val, yplot_val,
                                                 lambda x: inference(x, debug=False), num_classes=num_classes,
                                                 save_filepath=file_path, title='Precision %1.2f%%' % (avg_precision * 100.0))

    #-------------------------------------------------------------------------------------------------------------------
    # run keys projection for memories augmented neural networks
    if args.run_key_projection and isinstance(model_wrapper.model, models.model_mem.Model):
        experiment_folder = os.path.join(args.experiment_folder, 'keys_analysis', args.model_id)
        os.makedirs(experiment_folder, exist_ok=True)
        keys, vals, age = model_wrapper.model.get_memory_state(model_wrapper.sess)
        file_path = os.path.join(experiment_folder, 'keys_projection_' + str(iter))
        project_keys(keys, vals, age, num_components=2, num_classes=args.output_dim, file_path=file_path)

    #-------------------------------------------------------------------------------------------------------------------
    # run keys clustering for memories augmented neural networks
    if args.run_keys_clustering and isinstance(model_wrapper.model, models.model_mem.Model):
        file_path = os.path.join(experiment_folder, 'keys_clustering_{}_{}.html'.format(iter, avg_precision))
        key_stats = model_wrapper.run_key_analysis(dataset, file_path)

    print(colorize('[INFERENCE] Avg_precision: %f Metadata: %s' % (avg_precision, args.model_id), 'blue', bold=True))
    return avg_precision

#-----------------------------------------------------------------------------------------------------------------------
# Tests:
def run_memory_size(mem_size=1000):
    print('... Test different memories sizes')

    mem_sizes = args.mem_size
    if type(mem_sizes) == int:
        mem_sizes = [args.mem_size]

    output = []
    for mem_size in mem_sizes:
        print('------------MEMORY SIZE %d------------' % mem_size)
        avg_precision = []
        for i in range(args.k_fold):
            args.mem_size = mem_size

            model = get_model()
            model_api.initialize_model(model=model)
            train_model(info=str(mem_size))
            precision = report_inference(model_api, 0)
            avg_precision.append(precision)

        avg_precision = np.mean(avg_precision)
        output.append({'mem_size': mem_size, 'avg_precision': avg_precision})
    print('\nResults: ', output)
    return output

def run():
    #-------------------------------------------------------------------------------------------------------------------
    # instantiate model
    model_api.initialize_model(model=get_model())

    #-------------------------------------------------------------------------------------------------------------------
    # training model and report experiments on validation dataset every number of steps
    train_model()

    #-------------------------------------------------------------------------------------------------------------------
    # final inference on validation dataset. Useful when comparing multiple models
    y_preds = model_API.query_val(x_val, x_val_words, y_val)
    avg_precision = np.mean(np.equal(np.array(y_preds).ravel(), np.array(y_val).ravel()))
    output = []
    output.append({'mem_size': args.mem_size, 'precision': avg_precision})
    return output

#-----------------------------------------------------------------------------------------------------------------------
# Run:
if __name__ == '__main__':
    if not os.path.isdir(args.tensorboard_summary_folder):
        os.mkdir(args.tensorboard_summary_folder) 
    now = datetime.datetime.now()
    args.model_id = '%s_%d_%d.%d%s' % (args.model_name, now.day, now.hour, now.minute, args.info)
    run()
