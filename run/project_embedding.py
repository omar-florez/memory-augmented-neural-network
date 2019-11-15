import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
import csv
import os
import ipdb

summary_dir     = './temp/summaries/projector'
embedding_npy   = './temp/summaries/embedding_keys_70.npy'
label_npy       = './temp/summaries/embedding_labels_70.npy'
label_path      = './temp/summaries/projector/labels_70.tsv'

def run():
    #embedding_keys:    (1500, 128)
    embedding_keys = np.load(embedding_npy)
    #embedding_labels:  (1500, 1)
    embedding_labels = np.load(label_npy)
    with open(label_path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['i', 'class'])
        for ii, ll in enumerate(embedding_labels):
            writer.writerow([ii, ll[0]])

    embedding_var = tf.Variable(embedding_keys, name="validation_embedding_keys")
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #sess.run(embedding_var.initializer)
        saver.save(sess, os.path.join(summary_dir, "model.ckpt"), 0)

        config = projector.ProjectorConfig()

        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        embedding.metadata_path = '../../../' + label_path

        summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)
        projector.visualize_embeddings(summary_writer, config)
        summary_writer.close()

#python run/project_embedding.py
#tensorboard --logdir /Users/ost437/Documents/OneDrive/workspace/WorkspaceCapitalOne/learning_to_hash_labels/temp/summaries/projector/
if __name__ == '__main__':
    run()