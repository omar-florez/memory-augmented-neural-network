import ipdb
import os
import numpy as np
import random
import math
from collections import namedtuple
import time
import ipdb

class Tokens:
    ZEROPAD = ' '
    UNK = chr(255)
    BOW = chr(254)
    EOW = chr(253)
    BOS = chr(252)
    EOS = chr(251)
    USER = EOS

class LMDataPreprocessor(object):
    def __init__(self, data_dir, seq_len=32, wrd_len=65, sampleprob=1.0, iid=False, load_data=True, dialog_state=False):
        self.seq_len = seq_len
        self.wrd_len = wrd_len
        self.sampleprob = sampleprob
        self.dpath = data_dir
        self.iid = iid
        self.lm_fn = 'lm_data_iid.npz' if self.iid else 'lm_data.npz'
        #Assert is True if the below array is valid
        assert ([os.path.isfile(os.path.join(self.dpath, leaf)) for leaf in ['train.txt', 'valid.txt', 'test.txt',
                                                                'word2idx.txt', 'idx2word.txt', 'word2tok.txt',
                                                                'char2idx.txt', 'idx2char.txt']])

        self.wToid, self.idTow = self.__load_map(os.path.join(data_dir, 'word2idx.txt'), dialog_state=dialog_state)
        self.cToid, self.idToc = self.__load_map(os.path.join(data_dir, 'char2idx.txt'), dialog_state=dialog_state)
        self.wordvs = len(self.wToid)
        self.charvs = len(self.cToid)

        if load_data:
            self.c_inputs = list()
            self.w_inputs = list()
            self.w_targets = list()
            self.seq_lens = list()
            self.__load_data()
            self.len = len(self.w_targets)

    @staticmethod
    def __load_map(fname, cast_to_int=True, dialog_state=False):
        toidx = dict()
        idxto = dict()
        with open(fname, 'r', encoding="utf-8") as f:
            for line in f:
                if not line.split(): continue
                if dialog_state:
                    line = line.strip().split('\t')
                else:
                    line = line.split('\n')[0].split(' # ')[0].split('\t')

                key  = line[0][1:-1] if len(line[0])>1 and line[0][0]=="'" and line[0][-1]=="'" else line[0]
                val = line[1]
                if not dialog_state:
                    toidx[key[1: -1]] = int(val) if cast_to_int else val
                    idxto[int(val) if cast_to_int else val] = key[1: -1]
                else:
                    toidx[key] = int(val) if cast_to_int else val
                    idxto[int(val) if cast_to_int else val] = key
        return toidx, idxto

    def __load_map_dialog(fname, cast_to_int=True):
        toidx = dict()
        idxto = dict()
        with open(fname, 'r', encoding="utf-8") as f:
            for line in f:
                if not line.split(): continue
                line = line.strip().split('\t')
                key = line[0]
                val = line[1]
                toidx[key] = int(val) if cast_to_int else val
                idxto[int(val) if cast_to_int else val] = key
        return toidx, idxto

    def __load_data(self):
        if os.path.isfile(os.path.join(self.dpath, self.lm_fn)):
            data = np.load(os.path.join(self.dpath, self.lm_fn))
            self.c_inputs = data['c_inputs']
            self.w_inputs = data['w_inputs']
            self.w_targets = data['word_targets']
            self.seq_lens = data['seq_lens']
            del data
        else:
            t0 = time.time()
            self.__generate_dataset()

    def cToids(self, utt):
        c_seq = np.zeros(shape=[self.seq_len, self.wrd_len], dtype='int8')
        for i, wrd in enumerate(utt):
            if len(wrd) > self.wrd_len: wrd = wrd[:self.wrd_len]
            c_seq[i, :len(wrd)] = [self.cToid[char] if char in self.cToid else 0 for char in wrd]
        return c_seq

    def wToids(self, utt):
        wrd_seq = np.zeros(shape=[self.seq_len], dtype='int32')
        for i, wrd in enumerate(utt):
            wrd_seq[i] = self.wToid[wrd] if wrd in self.wToid else 0
        return wrd_seq

    def preprocess(self, utt):
        utt = remap_product(utt)
        utt = remap_emoji(utt)
        return utt

    def __generate_dataset(self):
        for fn in [os.path.join(self.dpath, leaf) for leaf in ['train.txt', 'valid.txt', 'test.txt']]:
            with open(fn, 'r') as f:
                for line in f:
                    if not line.split(): continue
                    if not self.iid:
                        utterance = sanitizer.sanitize(line,
                                                 use_dictionary=False,
                                                 remove_invalid=False,
                                                 twitter=False,
                                                 user_function=lambda utt: self.preprocess(line))
                        if not utterance: continue
                        utterance = utterance.strip('\t').strip('\n').split(' ')[:-1]
                        if len(utterance) > self.seq_len: utterance = utterance[:self.seq_len]
                        seq_len = len(utterance) - 1
                        words = self.wToids(utterance[1:])
                        chars = self.cToids(utterance[1:])
                        targs = np.concatenate((words[1:], np.array([0]))).astype(dtype='int32')

                        self.c_inputs.append(chars)
                        self.w_inputs.append(words)
                        self.w_targets.append(targs)
                        self.seq_lens.append(seq_len)
                    else:
                        if np.random.random() <= 22 * self.sampleprob:
                            if not utterance: continue
                            utterance = utterance.strip('\t').strip('\n').split(' ')
                            if len(utterance) > self.seq_len: utterance = utterance[:self.seq_len]
                            words = self.wToids(utterance)
                            for idx in range(len(words) - 1):
                                if np.random.random() <= float(1 / 22):
                                    chars_ = self.cToids(utterance[:idx + 1])
                                    words_ = self.wToids(utterance[:idx + 1])
                                    targs = words_[idx + 1]
                                    self.c_inputs.append(chars_)
                                    self.w_inputs.append(words_)
                                    self.w_targets.append(targs)
                                    self.seq_lens.append(idx + 1)

        self.c_inputs = np.array(self.c_inputs, dtype='int8')
        self.w_inputs = np.array(self.w_inputs, dtype='int32')
        self.w_targets = np.array(self.w_targets, dtype='int32') if not self.iid else np.array(self.w_targets, dtype='int32').reshape([-1, 1])
        self.seq_lens = np.array(self.seq_lens, dtype='int8')

        np.savez_compressed(file=os.path.join(self.dpath, self.lm_fn),
                            c_inputs=self.c_inputs,
                            w_inputs=self.w_inputs,
                            word_targets=self.w_targets,
                            seq_lens=self.seq_lens)

class IMDataPreprocessor(object):
    def __init__(self, intent_dir, filename, lm_dir, dialog_state=False):
        self.dpath = intent_dir
        self.filename = filename
        self.DL = LMDataPreprocessor(data_dir=lm_dir, load_data=False, dialog_state=dialog_state)
        self.len = None
        self.lToid = dict()
        self.idTol = dict()
        self.c_inputs = list()
        self.w_inputs = list()
        self.labels = list()
        self.seq_lens = list()
        self.__load_labels()
        self.label_counts = [0 for _ in self.idTol]
        self.__load_data()
        self.labelTodataIdxs = self.__load_label_idxs()
        self.class_wts = [i / sum(self.label_counts) for i in self.label_counts]

    def __load_labels(self):
        with open(os.path.join(self.dpath, 'label_map.txt'), 'r') as f:
            for line in f:
                if not line.split(): continue
                wrd = line.split(' = ')[0]
                label = int(line.split(' = ')[1])
                self.lToid[wrd] = label
                self.idTol[label] = wrd

    def __load_label_idxs(self):
        mp = {key: [] for key in self.idTol}
        for i in range(self.labels.shape[0]):
            mp[self.labels[i, 0]].append(i)
        return mp

    def __load_data(self):
        file_path = os.path.join(self.dpath, '{}.npz'.format(self.filename))
        if os.path.isfile(file_path):
            data = np.load(file_path)
            self.c_inputs = data['char_inputs']
            self.w_inputs = data['word_inputs']
            self.labels = data['intents']
            self.seq_lens = data['seq_lengths']
            self.label_counts = list(data['class_counts'])
            del data
        else:
            t0 = time.time()
            self.__generate_dataset()
        self.len = self.w_inputs.shape[0]

    def __generate_dataset(self):
        for fn in [os.path.join(self.dpath, leaf) for leaf in [self.filename]]:
            with open(fn, 'r', encoding='utf-8') as f:
                for line in f:
                    # Skip blank lines
                    if not line.split(): continue
                    m = line.split('\n')[0].split('\t')
                    utt = m[0]
                    intent = self.lToid[m[1]]
                    if not utt: continue
                    if len(utt) > self.DL.seq_len: utt = utt[:self.DL.seq_len]
                    words = self.DL.wToids(utt.split(' '))
                    chars = self.DL.cToids(utt.split(' '))
                    seq_len = len(utt.split(' '))
                    self.c_inputs.append(chars)
                    self.w_inputs.append(words)
                    self.labels.append(intent)
                    self.seq_lens.append(seq_len)
                    self.label_counts[intent] += 1

        # Convert words, chars, labels to np.arrays
        self.c_inputs = np.array(self.c_inputs)
        self.w_inputs = np.array(self.w_inputs)
        self.labels = np.array(self.labels).reshape([-1, 1])
        self.seq_lens = np.array(self.seq_lens)
        self.label_counts = np.array(self.label_counts)

        self.label_counts = list(self.label_counts)
        np.savez_compressed(file=os.path.join(self.dpath, self.filename),
                            c_inputs=self.c_inputs,
                            w_inputs=self.w_inputs,
                            labels = self.labels,
                            class_counts=self.label_counts,
                            #word_targets=self.word_targets,
                            seq_lens=self.seq_lens)




