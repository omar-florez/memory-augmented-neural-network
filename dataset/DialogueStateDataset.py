'''
Dataset wrapper to contain data about real labels
@author: Omar U. Florez
'''

import numpy as np
import random
from mutils.preprocessor import IMDataPreprocessor as IDL
#from mutils.plot_utils import save_histogram_figure
import collections
import ipdb
import os
from sklearn.utils import shuffle

class DialogueStateDataset():
    def __init__(self, data_dir=None):
        self.data_dir = data_dir
        self.build_dataset()

    def build_dataset(self, matrix_format=True):
        if self.data_dir:
            MAPS_PATH = os.path.join(self.data_dir, 'dialog_state', 'lm')
            TOKENS_PATH = os.path.join(self.data_dir, 'dialog_state', 'tokens')

        idl = IDL(intent_dir=TOKENS_PATH, filename='train.tsv', lm_dir=MAPS_PATH, dialog_state=True)
        idl_val = IDL(intent_dir=TOKENS_PATH, filename='valid.tsv', lm_dir=MAPS_PATH, dialog_state=True)

        # ---------------------------------------------------------------------------------------------------------------
        # Training dataset:
        # ---------------------------------------------------------------------------------------------------------------
        charvs = len(idl.DL.idToc)
        chardim = 16
        wordvs = len(idl.DL.idTow)
        self.seq_max_len = 32

        # to remove empty words (which also contains empty character-based observations)
        rows, cols, depth = idl.c_inputs.shape
        empty_chars_idx = np.where(idl.c_inputs.reshape([-1, cols * depth]).sum(axis=1) == 0)[0]
        empty_words_idx = np.where(idl.w_inputs.sum(axis=1) == 0)[0]

        # IDL.w_inputs.shape = (20438, 32) = (n, seq_len)
        n = idl.len
        indices = list(range(n))  # 20438 observations (20K)
        indices = [idx for idx in indices if idx not in empty_words_idx]  # 20057 observartions
        n = len(indices)
        np.random.shuffle(indices)

        cv_split = (1, 0, 0)
        nTr = int(cv_split[0] * n)  # nTr=14306 nTe=3067 nVal=3065
        nVal = int(cv_split[1] * n)
        nTe = n - nTr - nVal

        idxTr = indices[:nTr]  # 14306 training examples
        idxVal = indices[nTr:nTr + nVal]  # 3065 validation examples
        idxTe = indices[-nTe:]

        # ---------------------------------------------------------------------------------------------------------------
        # Validation dataset:
        # ---------------------------------------------------------------------------------------------------------------
        rows, cols, depth = idl_val.c_inputs.shape
        empty_chars_idx = np.where(idl_val.c_inputs.reshape([-1, cols * depth]).sum(axis=1) == 0)[0]
        empty_words_idx = np.where(idl_val.w_inputs.sum(axis=1) == 0)[0]

        n = idl_val.len
        indices = list(range(n))  # 20438 observations (20K)
        indices = [idx for idx in indices if idx not in empty_words_idx]  # 20057 observartions
        n = len(indices)
        np.random.shuffle(indices)

        cv_split = (1, 0, 0)
        nTr = int(cv_split[0] * n)  # nTr=14306 nTe=3067 nVal=3065
        nVal = int(cv_split[1] * n)
        nTe = n - nTr - nVal

        idxTr_val = indices[:nTr]  # 14306 training examples
        idxVal_val = indices[nTr:nTr + nVal]  # 3065 validation examples
        idxTe_val = indices[-nTe:]  # 3067 testing examples

        # Character-based input:
        if matrix_format:
            # input data: [batch_size, vocab_size*seq_max_len] [256, 2080]
            self.x_train, self.seqlen_train, self.y_train = self.convert_to_matrix(idl.c_inputs[idxTr],
                                                                                   idl.labels[idxTr])
            self.x_test, self.seqlen_test, self.y_test = self.convert_to_matrix(idl_val.c_inputs[idxTe_val],
                                                                                idl_val.labels[idxTe_val])
            self.x_val, self.seqlen_val, self.y_val = self.convert_to_matrix(idl_val.c_inputs[idxTr_val],
                                                                             idl_val.labels[idxTr_val])
        else:
            # input data: [batch_size, seq_max_len, vocab_size] [256, 32, 65]
            self.x_train, self.seqlen_train, self.y_train = idl.c_inputs[idxTr], idl.seq_lens[idxTr], idl.labels[
                idxTr]
            self.x_test, self.seqlen_test, self.y_test = idl_val.c_inputs[idxTe_val], idl_val.seq_lens[idxTe_val], \
                                                         idl_val.labels[idxTe_val]
            self.x_val, self.seqlen_val, self.y_val = idl_val.c_inputs[idxTr_val], idl_val.seq_lens[idxTr_val], \
                                                      idl_val.labels[idxTr_val]

        # Word-based input:
        self.x_train_words, self.seqlen_train_words = idl.w_inputs[idxTr], idl.seq_lens[idxTr]
        self.x_test_words, self.seqlen_test_words = idl_val.w_inputs[idxTe_val], idl_val.seq_lens[idxTe_val]
        self.x_val_words, self.seqlen_val_words = idl_val.w_inputs[idxTr_val], idl_val.seq_lens[idxTr_val]

        self.idx_train, self.idx_test, self.idx_val = idxTr, idxTe_val, idxTr_val
        self.seq_max_len = np.max(self.seqlen_train)  # 32 characters
        self.seq_max_len_words = np.max(self.seqlen_train_words)  # 10 words

        self.idl = idl
        self.idl_val = idl_val
        self.vocab_size = self.get_vocab_size()  # 47 characters
        self.vocab_size_words = self.get_vocab_words_size()  # 80645 words
        # ipdb.set_trace()
        return

    #-------------------------------------------------------------------------------------------------------------------
    # API methods
    #-------------------------------------------------------------------------------------------------------------------

    # def get_train_test_val_data(self):
    #     vocab_size = self.vocab_size
    #     vocab_words_size = self.vocab_words_size
    #
    #     output = [self.x_train, self.y_train, self.seqlen_train,
    #               self.x_test, self.y_test, self.seqlen_test,
    #               self.x_val, self.y_val, self.seqlen_val,
    #               vocab_size, seq_max_len]
    #     return output

    def get_idx2char(self):
        return self.idl.DL.idx2char

    #dictionary of labels
    def get_label_idx2name(self):
        return self.idl.idx2intent

    def get_vocab_size(self):
        return self.idl.DL.charvs

    def get_vocab_words_size(self):
        return self.idl.DL.wordvs

    def get_number_classes(self):
        return len(np.unique(self.idl.labels))

    def get_seq_max_len(self):
        #this method assumes the training dataset is initially in tensor format:
        #       (batch_size, seq_len, vocab_size) = (14306, 32, 65)
        return self.x_train.shape[1]

    #-------------------------------------------------------------------------------------------------------------------
    def next_train(self, batch_size=32):
        assert len(self.idx_train) > batch_size
        np.random.shuffle(self.idx_train)
        x_batch = self.idl.c_inputs[self.idx_train[:batch_size]]
        y_batch = self.idl.labels[self.idx_train[:batch_size]]
        batch_seqlen = self.idl.seq_lens[self.idx_train[:batch_size]]
        #batch_seqlen = np.expand_dims(batch_seqlen, 1)
        return x_batch, y_batch.ravel(), batch_seqlen.ravel()

    def next_train_balanced(self, batch_size=32, width=10, matrix_format=True):
        '''
        Build a mini-batch by randomly picking a number of distinct classes ('width') from the training batch.
        This ensures we ended up with a balanced class distribution in each mini-batch
        :param batch_size: number of observations in the current mini-batch. Yann recommends 32, https://arxiv.org/abs/1804.07612
        :param width: number of distinct classes in the current minibatch
        :return: x, y, and sequence length minibatches
        '''
        assert len(self.idx_train) > batch_size
        #class ids are unique elements
        class_ids = list(self.idl.intent2dataIdxs.keys())
        class_ids_batch = random.sample(class_ids, width)

        #class_support: number of elements for class
        class_support = batch_size//width
        idx_train_batch = [np.random.choice(self.idl.intent2dataIdxs[class_id], class_support)
                           for class_id in class_ids_batch]
        idx_train_batch = np.array(idx_train_batch).ravel()

        remaind_len = batch_size - len(idx_train_batch)
        idx_remaind_batch = []
        class_ids_remain = np.random.choice(class_ids_batch, remaind_len)
        for class_id in class_ids_remain:
            idx_remaind_batch.append(np.random.choice(self.idl.intent2dataIdxs[class_id], 1))
        idx_remaind_batch = np.array(idx_remaind_batch).ravel()

        idx_train_batch = np.append(idx_remaind_batch, idx_train_batch)

        np.random.shuffle(idx_train_batch)

        #x_batch:               (256, 32, 65)   (batch, seq_len, word_len)
        #x_batch:               (256,)
        #batch_seqlen           (256,)          number of words per utterance
        x_batch = self.idl.c_inputs[idx_train_batch]
        y_batch = self.idl.labels[idx_train_batch].ravel()
        batch_seqlen = self.idl.seq_lens[idx_train_batch]

        if matrix_format:
            #x_batch_matrix:    (32, 2080)
            #y_batch:           (32,)
            x_batch, batch_seqlen, y_batch = self.convert_to_matrix(x_batch, y_batch)

        if len(x_batch) != len(y_batch):
            ipdb.set_trace()
        return x_batch, y_batch, batch_seqlen


    #===================================================================================================================
    # candidate method to generate mini-batches to train and test models
    def split_data(self, x, labels):
        x_train = x[:int(len(x) * 0.8)]
        y_train = labels[:int(len(x) * 0.8)]
        # x_train = [np.matrix(xx) for xx in x_train]
        # y_train = [[ll] for ll in y_train]

        x_val = x[int(len(x) * 0.8):]
        y_val = labels[int(len(x) * 0.8):]
        x_val = [np.matrix(xx) for xx in x_val]
        y_val = [[ll] for ll in y_val]
        return x_train, y_train, x_val, y_val

    def get_batch(self, x, y, batch_size=20, width=10, balanced=False):
        if not balanced:
            x, y = shuffle(x, y)
            return x[:batch_size], y[:batch_size]
        else:
            class_ids = set(y)
            num_classes = batch_size // width
            num_reminder = batch_size % width

            assert num_classes <= len(class_ids)
            chosen_class_ids = random.sample(class_ids, num_classes)

            class_dict = {}
            for xx, yy in zip(x, y):
                if yy not in class_dict:
                    class_dict[yy] = []
                class_dict[yy].append(xx)

            balanced_x = []
            balanced_y = []
            for class_id in chosen_class_ids:
                index = np.arange(len(class_dict[class_id]))
                index = np.random.choice(index, width)
                chosen_x = np.array(class_dict[class_id])[index]
                chosen_y = np.ones(width) * class_id

                balanced_x.extend(chosen_x)
                balanced_y.extend(chosen_y)

            remainder_class_ids = np.random.choice(chosen_class_ids, num_reminder)
            for class_id in remainder_class_ids:
                index = np.random.randint(len(class_dict[class_id]))
                balanced_x.append(class_dict[class_id][index])
                balanced_y.append(class_id)
            return np.array(balanced_x), np.array(balanced_y)
        return None, None

    def get_batch_chars_words(self, mode='training', batch_size=20):
        assert mode in ("training", "testing", "validation"), "Dataset type not recognized"
        if mode == "training":
            index = np.arange(len(self.x_train))
            x_chars, x_words, y = self.x_train, self.x_train_words, self.y_train
        elif mode == "testing":
            index = np.arange(len(self.x_test))
            x_chars, x_words, y = self.x_test, self.x_test_words, self.y_test
        elif mode == "validation":
            index = np.arange(len(self.x_val))
            x_chars, x_words, y = self.x_val, self.x_val_words, self.y_val

        index = shuffle(index)[:batch_size]
        x_chars, x_words, y = x_chars[index], x_words[index], y[index]
        return x_chars, x_words, y

    #-------------------------------------------------------------------------------------------------------------------

    def convert_to_matrix(self, x_batch, y_batch):
        ''''
        Convert from original format to matrix representation. E.g.,
            [batch_size, seq_len, word_len] -> [batch_size, seq_len]
        where 'seq_len' is the number of words and 'word_len' is the length of a single word
        :param x_batch: utterance or sequence of words in its original format: [seq_len, max_word_length] = [32, 65]
        :param y_batch: labels associated to each utterance
        '''
        y_batch = y_batch.ravel()
        batch_size, seq_len, word_len = x_batch.shape
        #x_batch_matrix: (14306, 2080)
        x_batch_matrix = np.zeros((batch_size, seq_len*word_len))
        valid_indices = []
        batch_seqlen = []
        for index, (sentence, label) in enumerate(zip(x_batch, y_batch)):
            #sentence: (32, 65), every row is a word of an utterance
            word_list = [x[x != 0] for x in sentence if np.any(x != 0)]
            #no more characters available
            if len(word_list)<1:
                continue
            #separate each word with an empy character: 0
            utterance = []
            for temp in word_list:
                utterance.append(temp)
                utterance.append([0])
            utterance = np.concatenate(utterance, axis=0)
            seq_len = max(len(utterance)-1,0)
            x_batch_matrix[index, :seq_len] = utterance[:seq_len]
            batch_seqlen.append(seq_len)
            valid_indices.append(index)

        batch_seqlen = np.array(batch_seqlen)
        x_batch_matrix = x_batch_matrix[valid_indices]
        batch_label = y_batch[valid_indices]
        return x_batch_matrix, batch_seqlen, batch_label

    def matrix_to_text(self, x_batch, char_mode=True):
        dictionary = self.idl.DL.idx2char if char_mode else self.idl.DL.idTow
        output = []
        for row_id in range(len(x_batch)):
            idx_sentence = x_batch[row_id]
            last_char = len(idx_sentence)
            for i in range(len(idx_sentence)-1):
                if idx_sentence[i] == 0 and idx_sentence[i+1] == 0:
                    last_char = i
                    break
            text_array = [dictionary[c] for c in idx_sentence[:last_char]]
            output.append("".join(text_array))
        return output


    def report_training_batches(self, batch_size = 32, number_batches = 1000):
        x_dataset, y_dataset_unbalanced = [], []
        x_dataset_balanced, y_dataset_balanced = [], []

        for i in range(number_batches):
            x_batch, y_batch, _ = self.next_train(batch_size)
            x_dataset.append(x_batch)
            y_dataset_unbalanced.append(y_batch)
            x_batch_balanced, y_batch_balanced, _ = self.next_train_balanced(batch_size)
            x_dataset_balanced.append(x_batch_balanced)
            y_dataset_balanced.append(y_batch_balanced)

        y_dataset_unbalanced = np.array(y_dataset_unbalanced).ravel()
        y_dataset_balanced = np.array(y_dataset_balanced).ravel()

        save_histogram_figure(collections.Counter(y_dataset_unbalanced), 'hist_train_unbalanced.png',
                              title='Training tokens dataset (%d)'%len(y_dataset_unbalanced))
        save_histogram_figure(collections.Counter(self.y_test), 'hist_test_unbalanced.png',
                              title='Testing tokens dataset (%d)' % len(self.y_test))
        save_histogram_figure(collections.Counter(self.y_val), 'hist_val_unbalanced.png',
                              title='Validation tokens dataset (%d)' % len(self.y_val))

        save_histogram_figure(collections.Counter(y_dataset_balanced), 'hist_train_balanced.png',
                              title='Training tokens dataset (%d)' % len(y_dataset_balanced))
        print('Finished reporting on training batches (%d)'%len(y_dataset_unbalanced))

