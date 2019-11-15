'''
Dataset wrapper to contain data about real labels
@author: Omar U. Florez
'''

import numpy as np
import random
import collections
import ipdb
import os

class SyntheticData():
    def __init__(self, num_classes=5, num_elements_per_class=20, std=0.5, seed=0):
        np.random.seed(seed)
        self.build_dataset(num_classes, num_elements_per_class, std)

    def build_dataset(self, num_classes, num_elements_per_class, std):
        x = []
        y = []
        #randomly generate observations for each class
        for i in range(num_classes):
            mean_x = np.random.randint(20)
            mean_y = np.random.randint(20)
            obs_x = np.random.normal(mean_x, std, num_elements_per_class)
            obs_y = np.random.normal(mean_y, std, num_elements_per_class)
            for j in range(num_elements_per_class):
                x.append([obs_x[j], obs_y[j]])
                y.append(i)

        x, y = np.array(x), np.array(y)
        n = len(x)
        cv_split = (0.7, 0.15, 0.15)
        nTr = int(cv_split[0] * n)
        nVal = int(cv_split[1] * n)
        nTe = n - nTr - nVal
        self.idx = list(range(n))
        np.random.shuffle(self.idx)
        idxTr = self.idx[:nTr]
        idxVal = self.idx[nTr:nTr + nVal]
        idxTe = self.idx[-nTe:]

        self.idx_train, self.idx_test, self.idx_val = idxTr, idxTe, idxVal
        self.x_train = x[idxTr]
        self.y_train = y[idxTr]
        self.seqlen_train = 1
        self.x_test = x[idxTe]
        self.y_test = y[idxTe].ravel()
        self.seqlen_test = 1
        self.x_val = x[idxVal]
        self.y_val = y[idxVal].ravel()
        self.seqlen_val = 1
        self.num_classes = len(np.unique(y))

        self.vocab_size_words = len(self.x_train)
        self.x_val_words = None
        return

    def get_vocab_words_size(self):
        return self.vocab_size_words

    #-------------------------------------------------------------------------------------------------------------------
    #API methods
    def get_test_val_data(self, matrix_format=True):
        output = [self.x_test, self.y_test, self.seqlen_test, self.x_val, self.y_val,
                  self.seqlen_val, self.get_vocab_size() , self.get_seq_max_len()]
        return output

    def get_train_test_val_data(self):
        output = [self.x_train, self.y_train, self.seqlen_train,
                  self.x_test, self.y_test, self.seqlen_test,
                  self.x_val, self.y_val, self.seqlen_val,
                  self.get_vocab_size(), self.get_seq_max_len()]
        return output

    def get_label_idx2name(self):
        return {idx : str(y[idx]) for idx in range(len(self.idx))}

    def get_vocab_size(self):
        return 1

    def get_number_classes(self):
        return self.num_classes

    def get_seq_max_len(self):
        return 1

    def next_train(self, batch_size=32):
        assert len(self.idx_train) > batch_size
        np.random.shuffle(self.idx_train)
        x_batch = self.x_train[:batch_size]
        y_batch = self.y_train[:batch_size]
        batch_seqlen = np.ones(batch_size)
        return x_batch, y_batch.ravel(), batch_seqlen.ravel()