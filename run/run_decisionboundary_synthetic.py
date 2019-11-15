import plotly.graph_objs as go
import numpy as np
import ipdb
import random

from dataset.SyntheticDataset import SyntheticData
from demo import loader
from mutils import plot_nn_utils
from sklearn.utils import shuffle

#-----------------------------------------------------------------------------------------------------------------------
# Global params:
# save_dir    = './temp/checkpoints_synthetic_demo/'
# model_name  = 'model.ckpt-99100'
num_classes = 133

#-----------------------------------------------------------------------------------------------------------------------
# Dataset:
std = 5
print('LOADING DATASET')

dataset = SyntheticData(num_classes=num_classes, num_elements_per_class=400, std=std/3.0, seed=0)
dataset_info = dataset.get_train_test_val_data()
x, labels = dataset_info[0], dataset_info[1]
train_data = dataset_info[0]
train_label = dataset_info[1]
train_seqlen = dataset_info[2]
test_data = dataset_info[3]
test_label = dataset_info[4]
test_seqlen = dataset_info[5]
val_data = dataset_info[6]
val_label = dataset_info[7]
val_seqlen = dataset_info[8]
vocab_size = dataset_info[9]
seq_max_len = dataset_info[10]

len_intersection_train_test = len(set(train_label.ravel()).intersection(set(test_label.ravel())))
len_intersection_train_val = len(set(train_label.ravel()).intersection(set(val_label.ravel())))
print('Size of class intersection between train and test: %d' %len_intersection_train_test)
print('Size of class intersection between train and validation: %d' % len_intersection_train_val)

#-----------------------------------------------------------------------------------------------------------------------
# Auxiliary methods:
def split_data_old(x):
    x_train = x[:int(len(x) * 0.8)]
    y_train = labels[:int(len(x) * 0.8)]
    x_val = x[int(len(x) * 0.8):]
    y_val = labels[int(len(x) * 0.8):]

    x_train = [np.matrix(xx) for xx in x_train]
    y_train = [[ll] for ll in y_train]
    x_val = [np.matrix(xx) for xx in x_val]
    y_val = [[ll] for ll in y_val]
    return x_train, y_train, x_val, y_val

def split_data(x):
    x_train = x[:int(len(x) * 0.8)]
    y_train = labels[:int(len(x) * 0.8)]
    #x_train = [np.matrix(xx) for xx in x_train]
    #y_train = [[ll] for ll in y_train]

    x_val = x[int(len(x) * 0.8):]
    y_val = labels[int(len(x) * 0.8):]
    x_val = [np.matrix(xx) for xx in x_val]
    y_val = [[ll] for ll in y_val]
    return x_train, y_train, x_val, y_val

def get_batch(x, y, batch_size=20):
    x, y = shuffle(x, y)
    return x[:batch_size], y[:batch_size],

x_train, y_train, x_val, y_val = split_data(x)


#-----------------------------------------------------------------------------------------------------------------------
# Experiment methods:
#-----------------------------------------------------------------------------------------------------------------------

def run_model(model_name = "man", run_plot_boundary = True):
    pdb.set_trace()
    use_lsh = True if model_name == "lsh" else False
    mloader = loader.Loader()
    mloader.initialize_model(embedder_algo='baseline', use_lsh=use_lsh)
    train_model(mloader, model_name=model_name)
    predict_model(mloader, run_plot_boundary=run_plot_boundary, model_name=model_name)

def run_average_precision(num_experiments=5):
    output = {}
    for model_name in ["lsh", "man"]:
        average_precision = 0
        use_lsh = True if model_name == "lsh" else False
        for i in range(num_experiments):
            mloader = loader.Loader()
            mloader.initialize_model(embedder_algo='mlp', use_lsh=use_lsh)
            mloader.clear_memory()

            train_model(mloader)
            precision = predict_model(mloader)
            average_precision += precision
        output[model_name] = average_precision/num_experiments
    print('[run_average_precision] output: ', output)
    return output

#-------------------------------------------------------------------------------------------
def train_model(mloader, model_name='lsh'):
    num_epochs = 2000
    loss_array = []
    precision_array = []
    for i in range(num_epochs):
        #sample a single batch:
        x, y = get_batch(x_train, y_train, batch_size=64)
        y_preds, losses = mloader.train([x], [y])

        avg_precision = np.mean(np.equal(np.array(y_preds).ravel(), np.array(y).ravel()))
        avg_loss = np.mean(losses)
        loss_array.append([i, avg_loss])
        precision_array.append([i, avg_precision])

        if i%50 == 0:
            print('[TRAINING] Average training precision at step %d: %f' %(i, avg_precision))
        if i%100 == 0:
            predict_model(mloader, run_plot_boundary=True, model_name='%s-%d'%(model_name, i))

    print('Training done')
    plot_nn_utils.plot_function(loss_array, save_filepath='./train_loss.png')
    plot_nn_utils.plot_function(precision_array, save_filepath='./train_precision.png')
    return avg_precision

def predict_model(mloader, run_plot_boundary=False, model_name=''):
    print('[predict_model]: mloader.query(x_val)')

    #inference = mloader.query_val
    #y_preds = inference(x_val, y_val)
    inference = mloader.query
    y_preds = mloader.query_val(x_val, y_val)

    avg_precision = np.mean(np.equal(np.array(y_preds).ravel(), np.array(y_val).ravel()))
    print('[INFERENCE] avg_precision: ', avg_precision)

    xplot_train = np.array(x_train).ravel().reshape(-1, 2)
    yplot_train = np.array(y_train).ravel()

    xplot_val = np.array(x_val).ravel().reshape(-1, 2)
    yplot_val = np.array(y_val).ravel()
    #print('Predictions: ', mloader.query([xplot_val]))

    if run_plot_boundary:
        file_name = './temp/figures/datasets_decision_boundaries_%s.png' % model_name
        print('[predict_model]: run_plot_boundary')
        # plot_nn_utils.plot_datasets_and_boundary(xplot_train, yplot_train, xplot_val, yplot_val,
        #                                          lambda x, y: inference(x, y, debug=False), num_classes=num_classes,
        #                                          save_filepath=file_name, title='Precision %1.2f%%' % (avg_precision * 100.0))

        plot_nn_utils.plot_datasets_and_boundary(xplot_train, yplot_train, xplot_val, yplot_val,
                                                 lambda x: inference(x, debug=False), num_classes=num_classes,
                                                 save_filepath=file_name, title='Precision %1.2f%%' % (avg_precision * 100.0))

    print('Avg_precision: ', avg_precision)
    print('Inference done')
    return avg_precision

if __name__ == "__main__":
    #run_model(model_name="man", run_plot_boundary=True)
    run_model(model_name="lsh", run_plot_boundary=True)
    #run_average_precision(num_experiments=3)
