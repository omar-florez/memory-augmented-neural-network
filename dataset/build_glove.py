"""Build an np.array from some glove file and some vocab file

You need to download `glove.840B.300d.txt` from
https://nlp.stanford.edu/projects/glove/ and you need to have built
your vocabulary first (Maybe using `build_vocab.py`)
"""

__author__ = "Guillaume Genthial"

from pathlib import Path

import numpy as np

from dataset.IntentDataset import labelsequenceData

run_local = True

if run_local:
    data_dir = './data'
else:
    data_dir = '/home/ubuntu/learning_to_hash_labels/data'


if __name__ == '__main__':
    dataset = labelsequenceData(data_dir=data_dir)
    glove_dim = 300

    # Load vocab
    word_to_idx = dataset.idl.DL.word2idx

    # Array of zeros
    size_vocab = dataset.get_vocab_words_size()
    #embeddings = np.zeros((size_vocab, glove_dim))
    embeddings = np.random.uniform(-1.0, 1.0, size=(size_vocab, glove_dim))

    # Get relevant glove vectors
    found = 0
    print('Reading GloVe file (may take a while)')
    with Path('./data/glove/glove.840B.300d.txt').open() as f:
        for line_idx, line in enumerate(f):
            if line_idx % 100000 == 0:
                print('- At line {}'.format(line_idx))
            line = line.strip().split()
            if len(line) != 300 + 1:
                continue
            word = line[0]
            embedding = line[1:]
            if word in word_to_idx:
                found += 1
                word_idx = word_to_idx[word]
                embeddings[word_idx] = embedding

    # Done. Found 53,549 vectors for 80,645 words
    print('- done. Found {} vectors for {} words'.format(found, size_vocab))

    # Save np.array to file
    np.savez_compressed('./data/glove/glove_labels_dataset.npz', embeddings=embeddings)
