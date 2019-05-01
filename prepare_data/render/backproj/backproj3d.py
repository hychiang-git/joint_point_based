import sys
import os
import math
import time
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
from collections import Counter
from PIL import Image

import scannet_dataset
from utils import eval_utils
from utils import scannet_util 
g_label_ids = scannet_util.g_label_ids

BATCH_SIZE = 1
NUM_CLASSES = 21
FRAME_SKIP = 1
IGNORE_LABEL = 0
DATA_DIR = None
SPLIT = None
LOG_FOUT = None 

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def build_graph():
    image = tf.placeholder(dtype=tf.float32, shape=(None, 480, 640, 3))
    pixel_label = tf.placeholder(dtype=tf.int32, shape=(None, 480, 640, 1))
    pixel_weight = tf.placeholder(dtype=tf.float32, shape=(None, 480, 640, 1))
    pixel_label_onehot = tf.one_hot(pixel_label, NUM_CLASSES, axis=3)

    ops = {
        'image':image,
        'pixel_label':pixel_label,
        'pixel_weight':pixel_weight,
        'pixel_label_onehot':pixel_label_onehot,
    }
    return ops


def backproj_label(sess, ops):
    print('***** ', SPLIT)
    val_set = scannet_dataset.ScannetDatasetVal( \
        root=DATA_DIR,
        num_classes=NUM_CLASSES,
        split=SPLIT,
        get_depth=False,
        get_coord=False,
        get_pixmeta=True,
        point_from_depth=False,
        with_scene_point=True,
        frame_skip=FRAME_SKIP)
    dataset_recorder = eval_utils.get_recorder(NUM_CLASSES,
                                'pixel_backproj_3d')

    print("[INFO] Start Evaluation")
    num_scenes = len(val_set)
    for s in range(num_scenes):
        scene_data = val_set[s]

        pixel_scene_logit = np.zeros((len(scene_data['scene_point']), NUM_CLASSES))
        scene_recorder = eval_utils.get_recorder(
                             NUM_CLASSES,
                             'pixel_backproj_3d')

        num_view = scene_data['num_view'] 
        num_batch = math.ceil(float(num_view)/BATCH_SIZE)
        for bid in range(0, num_batch):
            start_vid = bid * BATCH_SIZE
            end_vid = start_vid + BATCH_SIZE 
            if end_vid > num_view:
                end_vid = num_view
            feed_dict = {
                ops['image']:scene_data['color_img_list'][start_vid:end_vid], 
                ops['pixel_label']:scene_data['pixel_label_list'][start_vid:end_vid], 
                ops['pixel_weight']:scene_data['pixel_weight_list'][start_vid:end_vid],
            }
            image, pixel_label, pixel_weight, pixel_label_onehot, \
            = sess.run([ 
                ops['image'], 
                ops['pixel_label'], 
                ops['pixel_weight'], 
                ops['pixel_label_onehot'], 
                ], 
                feed_dict=feed_dict)
            # squeeze
            pixel_weight = np.squeeze(pixel_weight)
            pixel_label = np.squeeze(pixel_label)
            pixel_label_onehot = np.squeeze(pixel_label_onehot)
            # stretch 
            pixel_weight = pixel_weight.reshape(-1)
            pixel_label = pixel_label.reshape(-1)
            pixel_label_onehot = pixel_label_onehot.reshape((-1, NUM_CLASSES))
            # pixel meta: x, y, z, v1, v2, v3, v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z
            pixmeta = np.array(scene_data['pixel_meta_list'][start_vid:end_vid]).reshape(-1, 15)
            mesh_vertex, v1w, v2w, v3w = eval_utils.barycentric_weight(pixmeta)
            pixel_scene_logit[mesh_vertex[:,0]] += (v1w*pixel_label_onehot)
            pixel_scene_logit[mesh_vertex[:,1]] += (v2w*pixel_label_onehot)
            pixel_scene_logit[mesh_vertex[:,2]] += (v3w*pixel_label_onehot)
            print('\r {}/{}, {}, {}/{}' \
                .format(s, num_scenes, scene_data['scene_name'], end_vid, num_view), end='')
        # scene record
        eval_utils.record(
            NUM_CLASSES, scene_recorder,      \
            pixel_backproj_3d_label = scene_data['scene_point'][:, 10],           \
            pixel_backproj_3d_pred = np.argmax(pixel_scene_logit, -1), 
         )
        # dataset record
        eval_utils.record(
            NUM_CLASSES, dataset_recorder,      \
            pixel_backproj_3d_label = scene_data['scene_point'][:, 10],           \
            pixel_backproj_3d_pred = np.argmax(pixel_scene_logit, -1), 
         )

        eval_utils.evaluate_score(scene_recorder)
        log_string('\n----- {} score ----'.format(scene_data['scene_name']))
        eval_utils.log_score(scene_recorder, NUM_CLASSES, log_func=log_string)
    eval_utils.evaluate_score(dataset_recorder)
    log_string('\n----- Whole Dataset score ----')
    eval_utils.log_score(dataset_recorder, NUM_CLASSES, log_func=log_string, list_method=True)



def run():
    log_string('\n[INFO] '+str(datetime.now()))
    ops = build_graph()
    #Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    backproj_label(sess, ops)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root_dir",required=True,  help="data root dir")
    parser.add_argument("--split", default="",  help="data split, e.g. train or val")
    parser.add_argument("--frame_skip", type=int, default=1,  help="frame to skip, default is 1 (sample rate = 20)")
    args = parser.parse_args()
    DATA_DIR = args.data_root_dir 
    SPLIT = args.split
    FRAME_SKIP = args.frame_skip
    LOG_FOUT = open('backproj_{}_log_{}.txt'.format(SPLIT, FRAME_SKIP*20), 'w')
    
    print('***  DATA Dir: ', DATA_DIR)
    print('***  Split: ', SPLIT)
    print('***  FRAME SKIP: ', FRAME_SKIP)
    print('***  Start Back Projecting ***')
    run()
