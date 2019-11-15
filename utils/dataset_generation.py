'''
Using a pre-trained model and TF API to recognize objects
'''

#imports
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import ipdb
import os

from collections import defaultdict
from io import StringIO
from PIL import Image
import cv2

from utils import label_map_util
from utils import visualization_utils as vis_util
from utils import util

#environment setup
sys.path.append('.')

#variables
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'        #fast
#MODEL_NAME = 'faster_rcnn_resnet101_coco_11_06_2017'   #medium
MODEL_FILE = MODEL_NAME + '.tar.gz'

#frozen trained models
PATH_TO_CKPT = os.path.join('files', MODEL_NAME,  'frozen_inference_graph.pb')
#label for each box
PATH_TO_LABELS = os.path.join('files', 'mscoco_label_map.pbtxt')
NUM_CLASES = 90

PATH_TO_VISUAL_DIR = './data/BreakfastII_15fps_qvga_sync/P06/cam01'
PATH_TO_GENERATED_DATASET = './data/generated/BreakfastII_15fps_qvga_sync/'
VIDEO_FILES = [x for x in os.listdir(PATH_TO_VISUAL_DIR) if x.endswith('.avi')]
IMAGE_SIZE = (12, 8)

# load label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map,
                                                            max_num_classes=NUM_CLASES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def download_if_needed():
    #create folders
    if not os.path.exists('./files'):
        os.makedirs('./files')

    #download models
    if not os.path.exists('./files/' + MODEL_FILE):
        opener = urllib.request.URLopener()
        opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, './files/' + MODEL_FILE)
    print('trained models downloaded')

    #untar file
    if not os.path.exists('./files/' + MODEL_NAME):
        tar_file = tarfile.open('./files/' + MODEL_FILE)

        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, os.path.join(os.getcwd(), 'files'))
    print('tar file uncompressed')


def extract_objects_video(sess, file_path, entries, skip_frame_every=100, show_video=True):
    cap = cv2.VideoCapture(file_path)

    image_tensor = entries['image_tensor']
    detection_boxes = entries['detection_boxes']
    detection_scores = entries['detection_scores']
    detection_classes = entries['detection_classes']
    num_detections = entries['num_detections']

    index = 0
    all_blobs = {}
    while cap.isOpened():
        ret, image_np = cap.read()

        if index % skip_frame_every == 0:
            image_expanded = np.expand_dims(image_np, axis=0)
            if image_np is None:
                break

            (boxes, scores, classes, num) = sess.run([detection_boxes,
                                                      detection_scores,
                                                      detection_classes,
                                                      num_detections],
                                                     feed_dict={
                                                         image_tensor: image_expanded
                                                     })

            blobs = util.extract_regions(image_np,
                                         np.squeeze(boxes),
                                         np.squeeze(classes).astype(np.int32),
                                         np.squeeze(scores),
                                         category_index,
                                         use_normalized_coordinates=True,
                                         min_score_thresh=0.3)

            for k in blobs:
                key = str(k)
                if key not in all_blobs:
                    all_blobs[key] = []
                all_blobs[key] += blobs[key]

            if show_video:
                cv2.imshow('object_detection', cv2.resize(image_np, (800, 600)))
            print('index: ', index)
            if index > 50000:
                break
        index += 1

        if show_video:
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
    cap.release()
    return all_blobs

def run():
    #load graph
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            od_graph_def.ParseFromString(fid.read())
            tf.import_graph_def(od_graph_def, name='')

    with detection_graph.as_default():
        with tf.Session(graph = detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            entries = {'image_tensor': image_tensor,
                       'detection_boxes': detection_boxes,
                       'detection_scores': detection_scores,
                       'detection_classes': detection_classes,
                       'num_detections': num_detections}

            for file_name in VIDEO_FILES:
                print('video: %s'%file_name)
                person_id = PATH_TO_VISUAL_DIR.split('/')[-2]
                all_blobs = extract_objects_video(sess, os.path.join(PATH_TO_VISUAL_DIR, file_name),
                                                  entries, show_video=False)
                write_regions(os.path.join(PATH_TO_GENERATED_DATASET, person_id), all_blobs, alphabet_name=person_id)

            cv2.destroyAllWindows()

def write_regions(write_folder, all_blobs, alphabet_name='object'):
    if not os.path.exists(write_folder):
        os.makedirs(write_folder)

    for class_name in all_blobs:
        class_folder = os.path.join(write_folder, class_name)
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)

        index = len([x for x in os.listdir(class_folder) if x.endswith('.jpg')])
        for blob in all_blobs[class_name]:
            image = blob['image']
            image.save(os.path.join(class_folder, '%s_%.3f_%i.jpg' %(alphabet_name, blob['score'], index)))
            index += 1

if '__main__' == __name__:
    download_if_needed()
    run()










