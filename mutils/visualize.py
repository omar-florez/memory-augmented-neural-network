'''
Code to contain the logic to visualize different components in the self-attention LSTM code
@author: Omar U. Florez
'''

from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import matplotlib.cm as cm
import os

import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
import csv
import os
import ipdb
import json


class Visualize():
    def __init__(self):
        return

    def build_embedding_projection(self, embedding, labels, projector_path, dict_labels=None):
        # Create labeled file
        label_path = os.path.join(projector_path, 'labels.tsv')
        with open(label_path, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['i', 'class'])
            for ii, ll in enumerate(labels):
                if dict_labels:
                    writer.writerow([ii, dict_labels[ll]])
                else:
                    writer.writerow([ii, ll])

        embedding_var = tf.Variable(embedding, name="embedding_projection")
        saver = tf.train.Saver()

        sess1 = tf.get_default_session()
        sess1.run(tf.global_variables_initializer())
        #sess.run(embedding_var.initializer)
        saver.save(sess1, os.path.join(projector_path, "model.ckpt"), 0)

        config = projector.ProjectorConfig()
        emb = config.embeddings.add()
        emb.tensor_name = embedding_var.name
        emb.metadata_path = '../../../' + label_path
        summary_writer = tf.summary.FileWriter(projector_path, sess1.graph)
        projector.visualize_embeddings(summary_writer, config)
        summary_writer.close()


    def build_char_lstm(self, embedding, input_data,
                        idx2char,
                        run_scale=True,
                        run_average_state=True,
                        save_path='./',
                        json_filename='cell_h_visualize.json'):
        '''
        Create JSON format needed to visualize inference LSTM given its internal memories.
        This visualize the h date within the memories one neuron of the memories state at a time

        Start web server with:

        :param embedding: the internal memories o output for each time step. If len(embedding.shape) > 2 is True, then
        its shape is: seq_len x [batch_size, state_size] because it comes from the LSTM memories. Else, its shape is
        [batch_size, seq_len] like the alpha attention value for each utterance.
        :param data: test data
        :param label: test labels
        :return:
        '''

        #embedding: seq_lenx(batch_size, state_size) -> [batch_size, seq_len, state_size]
        if type(embedding) is list:
            embedding = np.asarray(embedding).transpose([1, 0, 2])

        if run_average_state:
            # mean: compute average activation across all the 'state_size' dimensions. Each of the 'seq_len' LSTM units have a different
            # vectors because their values depend on the previous one
            # embedding:
            #embedding = np.ones_like(embedding) * np.mean(embedding, axis=2, keepdims=True)
            # embedding: [batch_size, seq_len, state_size] -> [batch_size, seq_len, 1]
            embedding = np.mean(embedding, axis=2, keepdims=True)

        if run_scale:
            embedding = (embedding-embedding.mean())/(np.sqrt(embedding.var())+0.00001)
            embedding /= max(abs(embedding.min()), abs(embedding.max()))

        input_data = input_data.astype(int)

        output = {}
        vectors = []
        utterances = ""

        #input_data:    (batch_size, seq_len)
        #embedding:     (batch_size, seq_len, state_size)
        for char_idxs, embs in zip(input_data, embedding):
            chars = [idx2char[c] for c in char_idxs]
            utterances += ''.join(chars)
            for emb in embs:
                vectors.append(emb.tolist())

        #len(vectors) == len(utterances) == batch_size x seq_len x [state_size]
        output['pca'] = vectors
        output['seq'] = utterances
        with open(os.path.join(save_path, json_filename), 'w') as f:
            json.dump(output, f)
        return

    def build_char_lstm_labeled(self, embedding, input_data, label_names, idx2char,
                        run_scale=True,
                        run_average_state=True,
                        save_path='./',
                        probs=None,
                        json_filename='cell_h_visualize.json'):
        '''
        Create JSON format needed to visualize inference LSTM given its internal memories.
        This visualize the h date within the memories one neuron of the memories state at a time

        Start web server with:

        :param embedding: the internal memories o output for each time step. If len(embedding.shape) > 2 is True, then
        its shape is: seq_len x [batch_size, state_size] because it comes from the LSTM memories. Else, its shape is
        [batch_size, seq_len] like the alpha attention value for each utterance.
        :param data: test data
        :param label: test labels
        :return:
        '''

        #embedding: seq_lenx(batch_size, state_size) -> [batch_size, seq_len, state_size]
        if type(embedding) is list:
            embedding = np.asarray(embedding).transpose([1, 0, 2])

        if run_average_state:
            # mean: compute average activation across all the 'state_size' dimensions. Each of the 'seq_len' LSTM units have a different
            # vectors because their values depend on the previous one
            # embedding:
            #embedding = np.ones_like(embedding) * np.mean(embedding, axis=2, keepdims=True)
            # embedding: [batch_size, seq_len, state_size] -> [batch_size, seq_len, 1]
            embedding = np.mean(embedding, axis=2, keepdims=True)

        if run_scale:
            embedding = (embedding-embedding.mean())/(np.sqrt(embedding.var())+0.00001)
            embedding /= max(abs(embedding.min()), abs(embedding.max()))

        input_data = input_data.astype(int)

        output = {}
        vectors = []
        utterances = ""
        ipdb.set_trace()
        #input_data:    (batch_size, seq_len)
        #embedding:     (batch_size, seq_len, state_size)
        for char_idxs, embs, label_name, i in zip(input_data, embedding, label_names, range(len(embedding))):
            chars = [idx2char[c] for c in char_idxs]
            label_name = ""
            if probs:
                label_name = label_names[i] + "(" + np.max(probs[i]) + ")"
            else:
                label_name = label_names[i]
            utterances += ''.join(chars) + '\t' + label_name + '\n'
            for emb in embs:
                vectors.append(emb.tolist())
            #for i in range(len(label_name)):
            #    vectors.append([-10000.0])

        #len(vectors) == len(utterances) == batch_size x seq_len x [state_size]
        output['pca'] = vectors
        output['seq'] = utterances

        with open(os.path.join(save_path, json_filename), 'w') as f:
            json.dump(output, f)
        return

    def build_char_lstm_labeled_comparison(self, embedding, input_data,
                                           predicted_labels,
                                           actual_labels,
                                           idx2char,
                                           run_scale=True,
                                           run_average_state=True,
                                           save_path='./',
                                           probs=None,
                                           json_filename='cell_h_visualize.json'):
        '''
        Create JSON format needed to visualize inference LSTM given its internal memories.
        This visualize the h date within the memories one neuron of the memories state at a time

        Start web server with:

        :param embedding: the internal memories o output for each time step. If len(embedding.shape) > 2 is True, then
        its shape is: seq_len x [batch_size, state_size] because it comes from the LSTM memories. Else, its shape is
        [batch_size, seq_len] like the alpha attention value for each utterance.
        :param data: test data
        :param label: test labels
        :return:
        '''

        #embedding: seq_lenx(batch_size, state_size) -> [batch_size, seq_len, state_size]
        if type(embedding) is list:
            embedding = np.asarray(embedding).transpose([1, 0, 2])

        if run_average_state:
            # mean: compute average activation across all the 'state_size' dimensions. Each of the 'seq_len' LSTM units have a different
            # vectors because their values depend on the previous one
            # embedding:
            #embedding = np.ones_like(embedding) * np.mean(embedding, axis=2, keepdims=True)
            # embedding: [batch_size, seq_len, state_size] -> [batch_size, seq_len, 1]
            embedding = np.mean(embedding, axis=2, keepdims=True)

        if run_scale:
            embedding = (embedding-embedding.mean())/(np.sqrt(embedding.var())+0.00001)
            embedding /= max(abs(embedding.min()), abs(embedding.max()))

        input_data = input_data.astype(int)

        output = {}
        vectors = []
        utterances = ""

        #input_data:    (batch_size, seq_len)
        #embedding:     (batch_size, seq_len, state_size)
        for char_idxs, embs, pred_label, actual_label, i in zip(input_data, embedding, predicted_labels, actual_labels, range(len(embedding))):
            chars = [idx2char[c] for c in char_idxs]
            if probs:
                pred_label = predicted_labels[i] + "(" + np.max(probs[i]) + ")"
            else:
                pred_label = predicted_labels[i]
            utterances += ''.join(chars) + '\t' + pred_label + '\t' +  actual_label + '\n'
            for emb in embs:
                vectors.append(emb.tolist())

        #len(vectors) == len(utterances) == batch_size x seq_len x [state_size]
        output['pca'] = vectors
        output['seq'] = utterances

        with open(os.path.join(save_path, json_filename), 'w') as f:
            json.dump(output, f)
        return

    #-------------------------------------------------------------------------------------------------------------------
    def visualize(self, embed, x_test, y_test):
        # two ways of visualization: scale to fit [0,1] scale
        # feat = embed - np.min(embed, 0)
        # feat /= np.max(feat, 0)

        # two ways of visualization: leave with original scale
        feat = embed
        ax_min = np.min(embed,0)
        ax_max = np.max(embed,0)
        ax_dist_sq = np.sum((ax_max-ax_min)**2)

        plt.figure()
        ax = plt.subplot(111)
        colormap = plt.get_cmap('tab10')
        shown_images = np.array([[1., 1.]])
        for i in range(feat.shape[0]):
            dist = np.sum((feat[i] - shown_images)**2, 1)
            if np.min(dist) < 3e-4*ax_dist_sq:   # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [feat[i]]]
            patch_to_color = np.expand_dims(x_test[i], -1)
            patch_to_color = np.tile(patch_to_color, (1, 1, 3))
            patch_to_color = (1-patch_to_color) * (1,1,1) + patch_to_color * colormap(y_test[i]/10.)[:3]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(patch_to_color, zoom=0.5, cmap=plt.cm.gray_r),
                xy=feat[i], frameon=False
            )
            ax.add_artist(imagebox)

        plt.axis([ax_min[0], ax_max[0], ax_min[1], ax_max[1]])
        # plt.xticks([]), plt.yticks([])
        plt.title('Embedding from the last layer of the network')
        plt.show()

    def visualize_noimage(self, embed,  y_test, file_path):

        # two ways of visualization: scale to fit [0,1] scale
        # feat = embed - np.min(embed, 0)
        # feat /= np.max(feat, 0)

        # two ways of visualization: leave with original scale
        feat = embed
        ax_min = np.min(embed,0)
        ax_max = np.max(embed,0)
        ax_dist_sq = np.sum((ax_max-ax_min)**2)

        plt.figure()
        ax = plt.subplot(111)
        colormap = plt.get_cmap('tab10')

        plt.axis([ax_min[0], ax_max[0], ax_min[1], ax_max[1]])
        # plt.xticks([]), plt.yticks([])
        plt.title('Learned embedding in the network')

        colors = cm.gist_ncar(np.linspace(0, 1, len(y_test)))
        #plt.plot(embed[:,0], embed[:,1],)
        for i in range(len(embed)):
            plt.scatter(embed[i][0], embed[i][1], color=colors[y_test[i]], s=8)

        plt.savefig(os.path.join(file_path+'.png'), dpi=600, format='png')#, bbox_inches='tight')
        plt.close()

# if __name__ == "__main__":
#
#     mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
#     x_test = mnist.test.images
#     y_test = mnist.test.labels
#     x_test = x_test.reshape([-1, 28, 28])
#
#     embed = np.fromfile('embed.txt', dtype=np.float32)
#     embed = embed.reshape([-1, 2])
#
#     visualize(embed, x_test, y_test)