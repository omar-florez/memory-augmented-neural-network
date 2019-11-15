import collections
import numpy as np

import six
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
import os
#import imageio
import shutil
import ipdb

from scipy.stats.kde import gaussian_kde
from numpy import linspace

def plot_distance_distributions(query_neigh_distances, file_path, neigh_size=10):
    plt.figure()
    fig, axes = plt.subplots(neigh_size, 1, figsize=(13, 20))
    fig.subplots_adjust(hspace=2, wspace=0.1)

    for k in range(neigh_size):
        distances = np.ravel([batch[:,:k+1] for batch in query_neigh_distances])

        ax = axes.flat[k]
        kde = gaussian_kde(distances)
        dist_space = linspace(min(distances), max(distances), 100)
        ax.plot(dist_space, kde(dist_space))
        title = 'Neighbor size: %d Mean: %f Variance: %f'%(k+1, np.mean(distances), np.var(distances))
        ax.set_title(title)
        ax.set_xlabel('Distance')
        ax.set_ylabel('Probability')
    plt.savefig(file_path)
    plt.close()

#-----------------------------------------------------------------------------------------------------------------------
#statistical functions
def entropy(x_continous):
    x_discrete = np.histogram(x_continous, bins=10, range=(0, 1.0), density=True)[0]
    c_normalized = x_discrete / float(np.sum(x_discrete))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    h = -sum(c_normalized * np.log(c_normalized))
    return h

def normalized_entropy(x_continous):
    x_discrete = np.histogram(x_continous, bins=10, range=(0, 1.0), density=True)[0]
    c_normalized = x_discrete / float(np.sum(x_discrete))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    h = -sum(c_normalized * np.log(c_normalized))/(len(c_normalized)+1e-5)
    return h

def L2_ranking(x_continous):
    #1.0 is perfect similarity and is assigned to first NN in target distribution
    target_distribution = np.zeros(len(x_continous))
    target_distribution[0] = 1.0
    loss = np.sqrt(np.square(target_distribution-np.array(x_continous)).sum())
    return loss

#-----------------------------------------------------------------------------------------------------------------------
#visualization
def plot_single_function(x, y, file_path, x_label='', y_label=''):
    plt.figure()
    plt.plot(x, y, 'bo', x, y, 'k')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(file_path)
    plt.close()

def plot_2_distance_distributions(query_neigh_distances_1,
                                  query_neigh_distances_2,
                                  file_path, neigh_size=10):
    plt.figure()
    fig, axes = plt.subplots(neigh_size, 1, figsize=(13, 20))
    fig.subplots_adjust(hspace=2, wspace=0.1)

    for k in range(neigh_size):
        distances_1 = np.ravel([batch[:,:k+1] for batch in query_neigh_distances_1])
        distances_2 = np.ravel([batch[:, :k + 1] for batch in query_neigh_distances_2])

        ax = axes.flat[k]
        kde_1 = gaussian_kde(distances_1)
        kde_2 = gaussian_kde(distances_2)
        dist_space_1 = linspace(min(distances_1), max(distances_1), 100)
        dist_space_2 = linspace(min(distances_2), max(distances_2), 100)
        ax.plot(dist_space_1, kde_1(dist_space_1), color='red')
        ax.plot(dist_space_2, kde_2(dist_space_2), color='blue')
        title = 'Neighbor size: %d | Mean: %f | Variance: %f'%(k+1, np.mean(distances_1), np.var(distances_1))
        title += '\nNeighbor size: %d | Mean: %f | Variance: %f'%(k+1, np.mean(distances_2), np.var(distances_2))
        ax.set_title(title)
        ax.set_xlabel('Distance')
        ax.set_ylabel('Probability')
    plt.savefig(file_path)
    plt.close()

def plot_2_averaged_similarity(query_neigh_distances_1, query_neigh_distances_2, file_path, neigh_size=10):
    plt.figure()
    fig, axes = plt.subplots(neigh_size, 1, figsize=(13, 25))
    fig.subplots_adjust(hspace=1.0, wspace=0.1)

    for k in range(neigh_size):
        distances_1 = np.concatenate([batch[:,:k+1] for batch in query_neigh_distances_1], axis=0)
        distances_2 = np.concatenate([batch[:, :k + 1] for batch in query_neigh_distances_2], axis=0)
        mean_distances_1 = np.mean(distances_1, axis=0)
        mean_distances_2 = np.mean(distances_2, axis=0)

        ax = axes.flat[k]
        ax.plot(np.arange(len(mean_distances_1))+1, mean_distances_1, '--ro', markersize=20)
        ax.plot(np.arange(len(mean_distances_2))+1, mean_distances_2, '--bo', markersize=20)
        title = 'Neighborhood size: %d | Mean: %f | Variance: %f'%(k+1, np.mean(mean_distances_1), np.var(mean_distances_1))
        title += '\nNeighbor size: %d | Mean: %f | Variance: %f'%(k+1, np.mean(mean_distances_2), np.var(mean_distances_2))

        ax.set_title(title)
        ax.set_xlabel('Neighborhood size')
        ax.set_ylabel('Average distance')
    plt.savefig(file_path)
    plt.close()

def plot_2_function_distributions(query_neigh_distances_1, query_neigh_distances_2,
                                  function, function_name, file_path, neigh_size=10):
    plt.figure()
    fig, axes = plt.subplots(neigh_size, 1, figsize=(13, 20))
    fig.subplots_adjust(hspace=2, wspace=0.1)

    for k in range(2,neigh_size):
        #consolidate all batches to one single array
        distances_1 = np.concatenate([batch[:,:k] for batch in query_neigh_distances_1], axis=0)
        distances_2 = np.concatenate([batch[:, :k] for batch in query_neigh_distances_2], axis=0)

        stats_1 = [function(x) for x in distances_1]
        stats_2 = [function(x) for x in distances_2]
        ax = axes.flat[k]
        kde_1 = gaussian_kde(stats_1)
        kde_2 = gaussian_kde(stats_2)
        dist_space_1 = linspace(min(stats_1), max(stats_1), 100)
        dist_space_2 = linspace(min(stats_2), max(stats_2), 100)
        ax.plot(dist_space_1, kde_1(dist_space_1), color='red')
        ax.plot(dist_space_2, kde_2(dist_space_2), color='blue')
        title = 'Neighbor size: %d | Mean: %f | Variance: %f'%(k+1, np.mean(stats_1), np.var(stats_1))
        title += '\nNeighbor size: %d | Mean: %f | Variance: %f'%(k+1, np.mean(stats_2), np.var(stats_2))
        ax.set_title(title)
        ax.set_xlabel(function_name)
        ax.set_ylabel('Probability')
    plt.savefig(file_path)
    plt.close()

#def smooth(values, factor):



def plot_tf_csv_files_from_folder(filepaths, labels, output_filepath='output_tensorflow_experiment.png'):
    import pandas as pd
    import ipdb
    dfs = [pd.read_csv(f) for f in filepaths]

    # data_df = pd.read_csv(file_name, sep=',')
    # corpus = data_df['observations']
    # emojis = data_df['emojis']
    k = np.arange(len(dfs)).astype(str)
    df = pd.concat([x.set_index('Step') for x in dfs], axis=1, join='inner', keys=k)

    number_experiments = len(dfs)
    plt.figure()
    fig, axes = plt.subplots(1, 1)
    fig.subplots_adjust(hspace=2, wspace=0.1)

    for i in range(number_experiments):
        x_label = 'steps'
        y_label = labels[i]
        steps = np.insert(df.index.values, 0, df.index.values[0])
        values = df[k[i]]['Value'].values
        values = np.insert(values, 0, values[0])
        values = pd.rolling_mean(values, 2)

        plt.plot(steps, values, label=labels[i])
        plt.xlabel(x_label)
        plt.ylabel(y_label)

    plt.legend(loc='best')

    # ax.set_title(title)
    # ax.set_xlabel('Distance')
    # ax.set_ylabel('Probability')
    plt.savefig(output_filepath)
    plt.close()
    ipdb.set_trace()




def make_all_gif(input_folder, save_filepath):
    time_per_step = 0.25
    for root, _, files in os.walk(os.path.join(input_folder, 'accuracy')):
        file_paths = [os.path.join(root, file) for file in files]
        #sorted by modified time
        file_paths = sorted(file_paths, key=lambda x: os.path.getmtime(x))
    file_names = [os.path.basename(file) for file in file_paths]

    episode_frames_accuracy = [imageio.imread(os.path.join(input_folder, 'accuracy',file_name)) for file_name in
                               file_names if file_name.endswith('.png')]
    episode_frames_boundary = [imageio.imread(os.path.join(input_folder, 'boundary', file_name)) for file_name in
                               file_names if file_name.endswith('.png')]
    episode_frames_loss = [imageio.imread(os.path.join(input_folder, 'loss', file_name)) for file_name in
                           file_names if file_name.endswith('.png')]

    assert(len(episode_frames_accuracy)==len(episode_frames_boundary)==len(episode_frames_loss))

    episode_frames = []
    for i in range(len(episode_frames_accuracy)):
        plt.figure()
        fig, axes = plt.subplots(1, 3, figsize=(20,5))
        #fig.subplots_adjust(hspace=1, wspace=1)

        ax = axes.flat[0]
        ax.imshow(episode_frames_accuracy[i], interpolation='none')
        ax.set_axis_off()
        ax.set_aspect('equal')

        ax = axes.flat[1]
        ax.imshow(episode_frames_boundary[i], interpolation='none')
        ax.set_axis_off()
        ax.set_aspect('equal')

        ax = axes.flat[2]
        ax.imshow(episode_frames_loss[i], interpolation='none')
        ax.set_axis_off()
        ax.set_aspect('equal')

        fig.tight_layout()
        plt.suptitle('Step = %d' %i, fontsize=18)
        plt.axis('off')
        plt.savefig(os.path.join(input_folder, 'all', 'image_%d.png'%i), dpi = 200)
        plt.close()

        image = imageio.imread(os.path.join(input_folder, 'all', 'image_%d.png'%i))
        episode_frames.append(image)

    episode_frames = np.array(episode_frames)
    imageio.mimsave(save_filepath, episode_frames, duration=time_per_step)
