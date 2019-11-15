import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import collections
import os
import ipdb

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def save_histogram_figure(counter, file_path, title=None):
    histogram = collections.Counter(counter)
    objects = [str(key) for key in np.sort([*histogram])]
    performance = [histogram[key] for key in np.sort([*histogram])]

    #objects = ('Python', 'C++', 'Java', 'Perl', 'Scala', 'Lisp')
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    if title:
        plt.title(title)
    plt.xlabel('Intent Id')
    plt.ylabel('Frequency')
    plt.xticks(size=2)
    #plt.yticks(size=10)
    plt.savefig(os.path.join(file_path), dpi=600, format='png')#, bbox_inches='tight')
    plt.close()


def save_loss_figure(file_path):
    losses = np.load(file_path)
    x = [xx[0] for xx in losses]
    y = [xx[1] for xx in losses]

    N = 400
    y_sma = np.convolve(y, np.ones((N,)) / N, mode='valid')


    plt.plot(x, y, linewidth=.5)
    plt.xlabel('Steps', size=8)
    plt.ylabel('Loss value', size=8)
    plt.ylim(0, np.max(y))
    plt.savefig(os.path.join(file_path+'.png'), dpi=600, format='png', bbox_inches='tight')
    plt.close()

    plt.plot(x[:len(y_sma)], y_sma, linewidth=.5)
    plt.xlabel('Steps', size=8)
    plt.ylabel('Loss value', size=8)
    plt.ylim(3, np.max(y_sma))
    plt.savefig(os.path.join(file_path + '.sma.png'), dpi=600, format='png', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    save_loss_figure('losses.npy')