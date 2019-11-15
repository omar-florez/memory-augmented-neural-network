# Loading Plotting Utilities
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use('Agg')

#import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
import os
#import imageio
import shutil
import ipdb

def plot_xor():
    xx, yy = np.meshgrid(np.linspace(-3, 3, 50), np.linspace(-3, 3, 50))
    rng = np.random.RandomState(0)
    X = rng.randn(300, 2)
    y = np.array(np.logical_xor(X[:, 0] > 0, X[:, 1] > 0), dtype=int)

    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure(figsize=(10, 8))

    ax = plt.subplot(gs[0, 0])
    plt.plot(X[np.where(y == 0), 0], X[np.where(y == 0), 1], 'ro')
    plt.plot(X[np.where(y == 1), 0], X[np.where(y == 1), 1], 'bo')
    plt.title('XOR')
    plt.show()

def plot_2d_points(X, save_filepath=None, text=None):
    plt.scatter(X[:, 0], X[:, 1])
    plt.xlabel('X[0]')
    plt.ylabel('X[1]')
    if text:
        plt.text(-3.2, 3.3, text, fontsize=14)
    if save_filepath == None:
        plt.show()
    else:
        plt.savefig(save_filepath)
    plt.close()


def plot_decision_boundary(X, y_actual, inference, num_classes, save_filepath=None, text=None, title=None):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    h = 1
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole gid
    zz = inference(np.c_[xx.ravel(), yy.ravel()])
    zz = np.array(zz).ravel()
    zz = zz.reshape(xx.shape)

    # Plot the contour and training examples
    plt.figure()
    levels = np.arange(num_classes+1)
    #levels = np.linspace(0, 9, 10)
    #plt.contourf(xx, yy, zz, cmap=plt.cm.Paired)
    cs = plt.contourf(xx, yy, zz, levels=levels, cmap='RdYlBu', alpha=.75)
    #cont = plt.contourf(X, Y, Z, 8, alpha=.75, cmap='jet')

    C = plt.contour(xx, yy, zz, 16, colors='black')
    #plt.clabel(C, inline=1, fontsize=10)

    #plt.scatter(X[:, 0], X[:, 1], color=['black'])
    labels = [str(l[0]) for l in y_actual]
    print('labels: ', labels)
    for i in np.arange(len(labels)):
        plt.text(X[i, 0], X[i, 1], labels[i], fontsize=12)
    plt.xlabel('X[0]')
    plt.ylabel('X[1]')
    plt.colorbar(cs, format="%.2f", ticks=levels)

    if title:
        plt.title(title)

    if text:
        plt.text(-3.2, 3.3, text, fontsize=14)
    if save_filepath == None:
        plt.show()
    else:
        plt.savefig(save_filepath)
    plt.close()

def plot_datasets_and_boundary(x_train, y_train, x_val, y_val, inference, num_classes, save_filepath=None, text=None, title=None):
    plt.figure(figsize=(36, 18))
    plt.subplots_adjust(hspace=0.6)
    if title:
        plt.suptitle(title, fontsize=40)

    #-------------------------------------------------------------------------------------------
    # Plot training dataset
    plt.subplot(1, 2, 1)
    plt.xlabel('X[0]')
    plt.ylabel('X[1]')

    x_min, x_max = x_train[:, 0].min() - 1.0, x_train[:, 0].max() + 1.0
    y_min, y_max = x_train[:, 1].min() - 1.0, x_train[:, 1].max() + 1.0
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    labels = [str(l) for l in y_train]
    for i in np.arange(len(labels)):
        plt.text(x_train[i, 0], x_train[i, 1], labels[i], fontsize=12)

    #-------------------------------------------------------------------------------------------
    # Plot validation dataset
    plt.subplot(1, 2, 2)
    # Set min and max values and give it some padding
    x_min, x_max = x_val[:, 0].min() - 1.0, x_val[:, 0].max() + 1.0
    y_min, y_max = x_val[:, 1].min() - 1.0, x_val[:, 1].max() + 1.0

    h = 1
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole gid
    # input.shape:   (N, 2)
    input = np.c_[xx.ravel(), yy.ravel()]

    model_input = [np.matrix(ii) for ii in input]
    model_label = [[5] for ii in input]

    #zz = inference(model_input, model_label)
    zz = inference(model_input)

    zz = np.array(zz).ravel()
    zz = zz.reshape(xx.shape)

    # Plot the contour and training examples
    levels = np.arange(-1,10,1)
    #levels = np.linspace(0, 9, 10)
    #plt.contourf(xx, yy, zz, cmap=plt.cm.Paired)
    cs = plt.contourf(xx, yy, zz, levels=levels, cmap='RdYlBu', alpha=.75)
    #cont = plt.contourf(X, Y, Z, 8, alpha=.75, cmap='jet')

    #plot lines separating classes:
    ##C = plt.contour(xx, yy, zz, 16, colors='black')

    # plot labels
    labels = [str(l) for l in y_val]
    for i in np.arange(len(labels)):
        plt.text(x_val[i, 0], x_val[i, 1], labels[i], fontsize=12)
    plt.xlabel('X[0]')
    plt.ylabel('X[1]')
    plt.colorbar(cs, format="%.2f", ticks=levels)

    if text:
        #plt.xlim(2, 2)
        #plt.ylim(0, 4)
        plt.text(-3.2, 3.3, text, fontsize=14)
    if save_filepath == None:
        plt.show()
    else:
        plt.savefig(save_filepath)
    plt.close()

def plot_function(losses, save_filepath=None, ylabel=None, title=None):
    plt.figure()
    t = [x[0] for x in losses]
    loss = [x[1] for x in losses]

    plt.figure()
    plt.plot(t, loss, 'b')
    plt.xlabel('Batch #')
    plt.ylabel(ylabel if ylabel else '')
    if title:
        plt.title(title)

    if save_filepath == None:
        plt.show()
    else:
        plt.savefig(save_filepath)
    plt.close()

def plot_array_dic(array_dic,
                   keys,
                   save_filepath=None,
                   title=None,
                   y_is_percentage=False):
    plt.figure()
    x = [dic[keys[0]] for dic in array_dic]
    y = [dic[keys[1]] for dic in array_dic]

    plt.figure()
    plt.plot(x, y, '-b.')
    plt.xlabel(keys[0])
    plt.ylabel(keys[1])

    if y_is_percentage:
        plt.ylim(0, 1)

    if title:
        plt.title(title)

    if save_filepath == None:
        plt.show()
    else:
        plt.savefig(save_filepath)
    plt.close()

def make_gif(input_folder, save_filepath):
    episode_frames = []
    time_per_step = 0.25
    for root, _, files in os.walk(input_folder):
        file_paths = [os.path.join(root, file) for file in files]
        #sorted by modified time
        file_paths = sorted(file_paths, key=lambda x: os.path.getmtime(x))
        episode_frames = [imageio.imread(file_path) for file_path in file_paths if file_path.endswith('.png')]
    episode_frames = np.array(episode_frames)
    imageio.mimsave(save_filepath, episode_frames, duration=time_per_step)

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

def reset_folders():
    folders = [os.path.join('./scratch_mlp/plots', f) for f in ['accuracy', 'boundary', 'loss']]
    if not os.path.exists('./scratch_mlp/plots'):
        os.mkdir('./scratch_mlp/plots')
    for f in folders:
        if os.path.exists(f):
            shutil.rmtree(f)
        os.mkdir(f)

if __name__ == '__main__':
    #save_filepath = './scratch_mlp/plots/gif/boundary.gif'
    #make_gif('./scratch_mlp/plots/boundary/', save_filepath)
    #save_filepath = './scratch_mlp/plots/gif/loss.gif'
    #make_gif('./scratch_mlp/plots/loss/', save_filepath)
    #save_filepath = './scratch_mlp/plots/gif/accuracy.gif'
    #make_gif('./scratch_mlp/plots/accuracy/', save_filepath)

    input_folder = './scratch_mlp/plots/'
    save_filepath = './scratch_mlp/plots/gif/all.gif'
    make_all_gif(input_folder, save_filepath)
