# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================
"""Data loading and other utilities.
Use this file to first copy over and pre-process the Omniglot dataset.
Simply call
  python data_utils.py
"""

#import cPickle as pickle
import pickle
import logging
import os
import subprocess
import ipdb
import random
import matplotlib.pyplot as plt

import numpy as np
from scipy.misc import imresize
from scipy.misc import imrotate
from scipy.ndimage import imread
import tensorflow as tf

#MAIN_DIR = '../learning_to_remember_rare_events'
MAIN_DIR = './'
REPO_LOCATION = 'https://github.com/brendenlake/omniglot.git'
REPO_DIR = os.path.join(MAIN_DIR, 'data','omniglot')
DATA_DIR = os.path.join(REPO_DIR, 'python')
TRAIN_DIR = os.path.join(DATA_DIR, 'images_background')
TEST_DIR = os.path.join(DATA_DIR, 'images_evaluation')
DATA_FILE_FORMAT = os.path.join(MAIN_DIR, 'files', '%s_omni.pkl')

TRAIN_ROTATIONS = True  # augment training data with rotations
TEST_ROTATIONS = False  # augment testing data with rotations
IMAGE_ORIGINAL_SIZE = 105
IMAGE_NEW_SIZE = 28

#episode_length: len(x)
#batch_size: x[0]
def write_summaries(x, y, file_path):
  parent_path = os.path.dirname(file_path)
  if not os.path.exists(parent_path):
    os.makedirs(parent_path)

  label_episode={}
  label_batches=[]

  for i, batch in enumerate(x):
    label_batch = {}
    for yy in y[i]:
      label_episode[yy] = label_episode.get(yy, 0) + 1
      label_batch[yy] = label_batch.get(yy, 0) + 1
    label_batches.append(label_batch)

  #plot episode
  plt.figure(figsize=(18, 18))
  plt.subplots_adjust(hspace = 0.6)

  plt.subplot(len(label_batches) // 5 + 1, 1, 1)
  labels = label_episode.keys()
  counts = [label_episode[x] for x in labels]
  y_pos = np.arange(len(labels))
  plt.bar(y_pos, counts, align='center', alpha=0.5)
  plt.xticks(y_pos, labels, rotation=90, fontsize=10.5)
  plt.ylabel('Frequency', fontsize=10.5)
  plt.title('Episode summary for %d classes' % len(labels), fontsize=15)

  for i, label_batch in enumerate(label_batches):
    plt.subplot(len(label_batches)//5+1, 5, i+5+1)
    labels = label_batch.keys()
    counts = [label_batch[x] for x in labels]
    y_pos = np.arange(len(labels))
    plt.bar(y_pos, counts, align='center', alpha=0.5)
    plt.xticks(y_pos, labels, rotation=90, fontsize=10.5)
    plt.ylabel('Frequency', fontsize=7.5)
    plt.title('%d classes' % len(labels), fontsize=15)

  plt.savefig(file_path)
  plt.close('all')


def get_data_omniglot(main_dir=None):
  """Get data in form suitable for episodic training.
  Returns:
    Train and test data as dictionaries mapping
    label to list of examples.
  """
  if main_dir is not None:
    MAIN_DIR = main_dir
  else:
    MAIN_DIR = './'

  DATA_FILE_FORMAT = os.path.join(MAIN_DIR, 'files', '%s_omni.pkl')
  with tf.gfile.GFile(DATA_FILE_FORMAT % 'train', 'rb') as f:
    processed_train_data = pickle.load(f, encoding='bytes')
  with tf.gfile.GFile(DATA_FILE_FORMAT % 'test', 'rb') as f:
    processed_test_data = pickle.load(f, encoding='bytes')

  train_data = {}
  test_data = {}
  data_info={}

  #------------------------------------------------------------------------------
  #LABELS
  #len(np.unique(processed_train_data['labels'])):  3856 characters REPETEAD 4 TIMES
  #                                                 len(processed_train_data['labels'])/4/20 = 964
  #                                                 964+659 = 1623 charac
  # ipdb > pp
  # processed_train_data['info'][:10]
  # ['omniglot/python/images_background/Alphabet_of_the_Magi/character01/0709_01.png',
  #  'omniglot/python/images_background/Alphabet_of_the_Magi/character01/0709_01.png',
  #  'omniglot/python/images_background/Alphabet_of_the_Magi/character01/0709_01.png',
  #  'omniglot/python/images_background/Alphabet_of_the_Magi/character01/0709_01.png',
  #  'omniglot/python/images_background/Alphabet_of_the_Magi/character01/0709_02.png',
  #  'omniglot/python/images_background/Alphabet_of_the_Magi/character01/0709_02.png',
  #  'omniglot/python/images_background/Alphabet_of_the_Magi/character01/0709_02.png',
  #  'omniglot/python/images_background/Alphabet_of_the_Magi/character01/0709_02.png',
  #  'omniglot/python/images_background/Alphabet_of_the_Magi/character01/0709_03.png',
  #  'omniglot/python/images_background/Alphabet_of_the_Magi/character01/0709_03.png']
  # ipdb > processed_train_data['labels'][:10]
  # array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1], dtype=uint32)


  #len(np.unique(processed_test_data['labels'])):   659 characters
  #processed_train_data['labels']     array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
  #processed_test_data['labels']      array([3856, 3856, 3856, 3856, 3856, 3856,

  #train_data.keys.keys(): [0, 1, 2, 3, 4, 5, 6, 7, 8, ...
  # len(train_data.keys): 3856
  #test_data.keys.keys(): [3856, 3857, 3858, 3859, 3860, 3861, 3862, ...
  # len(test_data.keys): 659
  #total=4515

  #EACH CHARACTER IS REPEATED 20 TIMES FOR 20 PEOPLE
  #processed_test_data['info'][1]: 'omniglot/python/images_evaluation/Angelic/character01/0965_02.png'
  #processed_test_data['info'][2]: 'omniglot/python/images_evaluation/Angelic/character01/0965_03.png'
  #processed_test_data['info'][3]: 'omniglot/python/images_evaluation/Angelic/character01/0965_04.png'
  #processed_test_data['info'][20]: 'omniglot/python/images_evaluation/Angelic/character02/0966_01.png'
  for data, processed_data in zip([train_data, test_data],
                                  [processed_train_data, processed_test_data]):
    #ipdb.set_trace()
    for image, label, info in zip(processed_data[b'images'],
                                  processed_data[b'labels'],
                                  processed_data[b'info']):
      if label not in data:
        data[label] = []
        data_info[label] = []
      data[label].append(image.reshape([-1]).astype('float32'))
      data_info[label].append(info)

  #------------------------------------------------------------------------------
  #RESULT
  #processed_train_data: ['images', 'info', 'labels']
  #len(processed_train_data['images']):             77120 characters made by 20 people
  #len(processed_test_data['images']):              13180 characters made by 20 people


    intersection = set(train_data.keys()) & set(test_data.keys())

  assert not intersection, 'Train and test data intersect.'
  ok_num_examples = [len(ll) == 20 for _, ll in train_data.items()]
  assert all(ok_num_examples), 'Bad number of examples in train data.'
  ok_num_examples = [len(ll) == 20 for _, ll in test_data.items()]
  assert all(ok_num_examples), 'Bad number of examples in test data.'

  #--------------------------------------------------------------------------------
  #DATASETS: each label has 20 different characters each made by a different person
  #len(train_data): 3856
  #len(train_data[0]): 20 images
  #train_data[3855][0]: (784,) each image has this dimension (28x28)
  logging.info('Number of labels in train data: %d.', len(train_data))
  #659
  logging.info('Number of labels in test data: %d.', len(test_data))

  #NUMBER OF CHARACTERS OR CLASSES
  #len(train_data): 3856    <= 77120/20
  #len(test_data): 659      <= 13180/20
  return train_data, test_data, data_info

def get_data_breakfast(DATA_FILE_FORMAT):
  """Get data in form suitable for episodic training.
  Returns:
    Train and test data as dictionaries mapping
    label to list of examples.
  """
  with tf.gfile.GFile(DATA_FILE_FORMAT % 'train') as f:
    processed_train_data = pickle.load(f)
  with tf.gfile.GFile(DATA_FILE_FORMAT % 'test') as f:
    processed_test_data = pickle.load(f)

  train_data = {}
  test_data = {}
  data_info={}

  for data, processed_data in zip([train_data, test_data],
                                  [processed_train_data, processed_test_data]):
    for image, label, info in zip(processed_data['images'],
                                  processed_data['labels'],
                                  processed_data['info']):
      if label not in data:
        data[label] = []
        data_info[label] = []
      data[label].append(image.reshape([-1]).astype('float32'))
      data_info[label].append(info)

  intersection = set(train_data.keys()) & set(test_data.keys())
  assert not intersection, 'Train and test data intersect.'
  ok_num_examples = [len(ll) == 20 for _, ll in train_data.iteritems()]
  assert all(ok_num_examples), 'Bad number of examples in train data.'
  ok_num_examples = [len(ll) == 20 for _, ll in test_data.iteritems()]
  assert all(ok_num_examples), 'Bad number of examples in test data.'

  #--------------------------------------------------------------------------------
  #DATASETS: each label has 20 different characters each made by a different person
  #len(train_data): 3856
  #len(train_data[0]): 20 images
  #train_data[3855][0]: (784,) each image has this dimension (28x28)
  logging.info('Number of labels in train data: %d.', len(train_data))
  #659
  logging.info('Number of labels in test data: %d.', len(test_data))

  ipdb.set_trace()

  #NUMBER OF CHARACTERS OR CLASSES
  #len(train_data): 3856    <= 77120/20
  #len(test_data): 659      <= 13180/20
  return train_data, test_data

def crawl_directory(directory, augment_with_rotations=False,
                    first_label=0, image_format = '.png'):
  """Crawls data directory and returns stuff."""
  label_idx = first_label
  images = []
  labels = []
  info = []

  # traverse root directory
  for root, _, files in os.walk(directory):
    logging.info('Reading files from %s', root)
    fileflag = 0
    for file_name in files:
      if not file_name.endswith(image_format):
        continue
      full_file_name = os.path.join(root, file_name)
      img = imread(full_file_name, flatten=True)
      for i, angle in enumerate([0, 90, 180, 270]):
        if not augment_with_rotations and i > 0:
          break

        images.append(imrotate(img, angle))
        labels.append(label_idx + i)
        info.append(full_file_name)

      fileflag = 1

    if fileflag:
      label_idx += 4 if augment_with_rotations else 1

  return images, labels, info


def resize_images(images, new_width, new_height):
  """Resize images to new dimensions."""
  resized_images = np.zeros([images.shape[0], new_width, new_height],
                            dtype=np.float32)

  for i in range(images.shape[0]):
    resized_images[i, :, :] = imresize(images[i, :, :],
                                       [new_width, new_height],
                                       interp='bilinear',
                                       mode=None)
  return resized_images


def write_datafiles(directory, write_file,
                    resize=True, rotate=False,
                    new_width=IMAGE_NEW_SIZE, new_height=IMAGE_NEW_SIZE,
                    first_label=0):
  """Load and preprocess images from a directory and write them to a file.
  Args:
    directory: Directory of alphabet sub-directories.
    write_file: Filename to write to.
    resize: Whether to resize the images.
    rotate: Whether to augment the dataset with rotations.
    new_width: New resize width.
    new_height: New resize height.
    first_label: Label to start with.
  Returns:
    Number of new labels created.
  """

  # these are the default sizes for Omniglot:
  imgwidth = IMAGE_ORIGINAL_SIZE
  imgheight = IMAGE_ORIGINAL_SIZE

  logging.info('Reading the data.')
  images, labels, info = crawl_directory(directory,
                                         augment_with_rotations=rotate,
                                         first_label=first_label)

  images_np = np.zeros([len(images), imgwidth, imgheight], dtype=np.bool)
  labels_np = np.zeros([len(labels)], dtype=np.uint32)
  for i in xrange(len(images)):
    images_np[i, :, :] = images[i]
    labels_np[i] = labels[i]

  if resize:
    logging.info('Resizing images.')
    resized_images = resize_images(images_np, new_width, new_height)

    logging.info('Writing resized data in float32 format.')
    data = {'images': resized_images,
            'labels': labels_np,
            'info': info}
    with tf.gfile.GFile(write_file, 'w') as f:
      pickle.dump(data, f)
  else:
    logging.info('Writing original sized data in boolean format.')
    data = {'images': images_np,
            'labels': labels_np,
            'info': info}
    with tf.gfile.GFile(write_file, 'w') as f:
      pickle.dump(data, f)

  return len(np.unique(labels_np))

def process_observation(images, labels, resize=True, rotate=False,
                        new_width=IMAGE_NEW_SIZE, new_height=IMAGE_NEW_SIZE):
  imgwidth, imgheight = images[0].shape[:2]
  images_np = np.zeros([len(images), imgwidth, imgheight], dtype=np.bool)
  labels_np = np.zeros([len(labels)], dtype=np.uint32)
  for i in xrange(len(images)):
    images_np[:,i] = images[i]
    labels_np[:,i] = labels[i]

  if resize:
    resized_images = resize_images(images_np, new_width, new_height)

  return resized_images




def maybe_download_data():
  """Download Omniglot repo if it does not exist."""
  if os.path.exists(REPO_DIR):
    logging.info('It appears that Git repo already exists.')
  else:
    logging.info('It appears that Git repo does not exist.')
    logging.info('Cloning now.')

    subprocess.check_output('git clone %s' % REPO_LOCATION, shell=True)

  if os.path.exists(TRAIN_DIR):
    logging.info('It appears that train data has already been unzipped.')
  else:
    logging.info('It appears that train data has not been unzipped.')
    logging.info('Unzipping now.')

    subprocess.check_output('unzip %s.zip -d %s' % (TRAIN_DIR, DATA_DIR),
                            shell=True)

  if os.path.exists(TEST_DIR):
    logging.info('It appears that test data has already been unzipped.')
  else:
    logging.info('It appears that test data has not been unzipped.')
    logging.info('Unzipping now.')

    subprocess.check_output('unzip %s.zip -d %s' % (TEST_DIR, DATA_DIR),
                            shell=True)


def preprocess_omniglot():
  """Download and prepare raw Omniglot data.
  Downloads the data from GitHub if it does not exist.
  Then load the images, augment with rotations if desired.
  Resize the images and write them to a pickle file.
  """

  maybe_download_data()

  directory = TRAIN_DIR
  write_file = DATA_FILE_FORMAT % 'train'
  num_labels = write_datafiles(
      directory, write_file, resize=True, rotate=TRAIN_ROTATIONS,
      new_width=IMAGE_NEW_SIZE, new_height=IMAGE_NEW_SIZE)

  directory = TEST_DIR
  write_file = DATA_FILE_FORMAT % 'test'
  write_datafiles(directory, write_file, resize=True, rotate=TEST_ROTATIONS,
                  new_width=IMAGE_NEW_SIZE, new_height=IMAGE_NEW_SIZE,
                  first_label=num_labels)


def sample_episode_batch(data, episode_length, episode_width, batch_size):
  # np.unique([yy[0] for yy in y]):
  #       [1027, 2132, 3115, 3281, 3784]
  # np.sort(bb)
  #       [496, 496, 496, 496, 496, 496, 1383, 1383, 1383, 1383, 1383,
  #        1383, 2072, 2072, 2072, 2072, 2072, 2072, 2125, 2125, 2125, 2125,
  #        2125, 2125, 2563, 2563, 2563, 2563, 2563, 2563]
  #   episode_width*


  """Generates a random batch for training or validation.
  Structures each element of the batch as an 'episode'.
  Each episode contains episode_length examples and
  episode_width distinct labels.
  Args:
    data: A dictionary mapping label to list of examples.
    episode_length: Number of examples in each episode.
    episode_width: Distinct number of labels in each episode.
    batch_size: Batch size (number of episodes).
  Returns:
    A tuple (x, y) where x is a list of batches of examples
    with size episode_length and y is a list of batches of labels.
  """

  episodes_x = [[] for _ in range(episode_length)]
  episodes_y = [[] for _ in range(episode_length)]
  assert len(data) >= episode_width

  # len(keys): 3856
  keys = data.keys()
  for b in range(batch_size):
    # episode_labels: [842, 705, 572, 3808, 2224]
    episode_labels = random.sample(keys, episode_width)
    remainder = episode_length % episode_width
    remainders = [0] * (episode_width - remainder) + [1] * remainder

    # sample this amount of times: (episode_length - remainder) / episode_width)
    # but add 1 elements when remainders is different than zero
    episode_x = [random.sample(data[lab], r + (episode_length - remainder) / episode_width)
                 for lab, r in zip(episode_labels, remainders)]

    # (x, i, ii): <observation, character local index, observation local index>
    episode = sum([[(x, i, ii) for ii, x in enumerate(xx)]
                   for i, xx in enumerate(episode_x)], [])
    random.shuffle(episode)
    # Arrange episode so that each distinct label is seen before moving to
    # 2nd showing
    episode.sort(key=lambda elem: elem[2])
    assert len(episode) == episode_length

    # iterate over episode_length=30 episodes
    # this way we will uniquely see a label 'episode_width' number of times across different 'episode_width' number of batches
    for i in range(episode_length):
      episodes_x[i].append(episode[i][0])
      ##episodes_y[i].append(episode[i][1] + b * episode_width)     #???
      episodes_y[i].append(episode_labels[episode[i][1]])  # ???

  return ([np.array(xx).astype('float32') for xx in episodes_x],
          [np.array(yy).astype('int32') for yy in episodes_y])


# -------------------------------------------------------------------------------------------------------------------
# Data
def generate_continous_validation_dataset(valid_data):
    '''
    omniglot dataset contains 20 elements per class
    observations_per_class = 20
    episode_width = 100 (number of classes)
    batch_size = 1
    
    divide validation dataset in:
      - 50% unseen classes
      - 25% partially unseen classes 
      - 25% partially seen classes (used for training)
    '''
    validation_classes = np.array(valid_data.keys())
    fully_unseen_classes = validation_classes[len(validation_classes) // 2:]
    partially_unseen_classes = validation_classes[:len(validation_classes) // 2]

    # build fully unseen dataset: it has 329 classes with 20 observations
    fully_unseen_dataset = {label:valid_data[label] for label in fully_unseen_classes}

    #build partially unseen dataset: both have 329 classes with 10 observations each one
    partially_unseen_dataset = {label:valid_data[label][:len(valid_data[label])//2] for label in partially_unseen_classes}
    partially_seen_dataset = {label:valid_data[label][len(valid_data[label])//2:] for label in partially_unseen_classes}
    return fully_unseen_dataset, partially_unseen_dataset, partially_seen_dataset

#------------------------------------------------------------------------------------------------
def main():
  logging.basicConfig(level=logging.INFO)
  preprocess_omniglot()


if __name__ == '__main__':
  main()
  #train_data, valid_data, data_info = get_data_omniglot()
  #ipdb.set_trace()
  #print('.')
