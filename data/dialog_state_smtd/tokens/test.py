import ipdb
import numpy as np

if __name__ == '__main__':
    filename = 'train.tsv'
    labels = []
    with open(filename) as f:
        for i, line in enumerate(f):
            if i < 1:
                continue
            labels.append(line.split('\t')[2].rstrip('\r\n'))
    labels = np.unique(labels)

    with open('mylabels.txt', 'w') as f:
        for i, label in enumerate(labels):
            f.write('{} = {}\n'.format(label, i))

    print('')
