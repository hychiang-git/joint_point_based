import sys
import os
import time
import math
import tensorflow as tf
import numpy as np
from datetime import datetime

from core import model 
from core import eval_utils 
from core import scannet_dataset
from utils import utils
from utils import scannet_util 
g_label_ids = scannet_util.g_label_ids

slim = tf.contrib.slim
# model snapshots dir
SNAPSHOT_DIR = os.path.join('log', 'train_log', 'snapshots')
MODEL_NAME = 'model'
# dataset info
BATCH_SIZE = 16
NUM_CLASSES = 21
IGNORE_LABEL = 0
IMAGE_SIZE = (480, 640)
COORD_SIZE = (120, 160)
DATA_PATH = os.path.join('/tmp3/hychiang/scannetv2_preprocess/')
VAL_DATASET_WHOLE_SCENE = scannet_dataset.ScannetDatasetVal( \
        root=DATA_PATH,
        num_classes=NUM_CLASSES,
        split='val',
        get_depth=False,
        get_coord=False,
        get_pixmeta=False,
        point_from_depth=False,
        with_scene_point=False,
        frame_skip=5)
# logging
LOG_DIR = os.path.join('log', 'val_log')
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_val.txt'), 'w')
# logging funcion
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def build_graph():
    is_training = False
    # placeholder 
    color_img = tf.placeholder(dtype=tf.float32, shape=(None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    pixel_label = tf.placeholder(dtype=tf.int32, shape=(None, IMAGE_SIZE[0], IMAGE_SIZE[1], 1))
    pixel_weight = tf.placeholder(dtype=tf.float32, shape=(None, IMAGE_SIZE[0], IMAGE_SIZE[1], 1))
    coord_img = tf.placeholder(dtype=tf.float32, shape=(None, IMAGE_SIZE[0], IMAGE_SIZE[1], 6))
    coord_label = tf.placeholder(dtype=tf.int32, shape=(None, IMAGE_SIZE[0], IMAGE_SIZE[1], 1))
    coord_weight = tf.placeholder(dtype=tf.float32, shape=(None, IMAGE_SIZE[0], IMAGE_SIZE[1], 1))
    # jointly encoder
    with tf.name_scope('joint_encoder') as scope:
        feature, deeplab_end_point = \
            model.joint_encoder(
                images=color_img,
                coords=coord_img[:, :, :, 0:3],
                batch_size = 1,
                output_stride=16,
                depth_init_size=COORD_SIZE,
                bn_decay=None,
                is_training=is_training,
                fine_tune_batch_norm=False)
    with tf.name_scope('deeplab_aspp') as scope:
        feature = model.deeplab_aspp_module(
            features=feature,
            image_size=IMAGE_SIZE,
            output_stride=16,
            is_training=is_training,
            fine_tune_batch_norm=False)
    with tf.name_scope('deeplab_decoder') as scope:
        feature = model.deeplab_decoder(
            features=feature,
            end_points=deeplab_end_point,
            decoder_output_stride=4,
            image_size=IMAGE_SIZE,
            is_training=is_training,
            fine_tune_batch_norm=False)
    with tf.name_scope('deeplab_output_layer') as scope:
        deeplab_logit = model.deeplab_output_layer(
            features=feature,
            num_classes=NUM_CLASSES)
    with tf.name_scope('deeplab_loss') as scope:
        deeplab_logit = tf.image.resize_bilinear(
            deeplab_logit, 
            tf.shape(pixel_label)[1:3], 
            align_corners=True)
        deeplab_loss = model.get_loss(
            logits=deeplab_logit,
            labels=pixel_label,
            num_classes=NUM_CLASSES,
            ignore_label=IGNORE_LABEL,
            weights=pixel_weight)

    deeplab_pred = tf.argmax(deeplab_logit, axis=-1) 
    log_string('[INFO] Deeplab Logit:\n'+str(deeplab_logit)+'\n--\n')
    log_string('[INFO] Deeplab Prediction:\n'+str(deeplab_pred)+'\n--\n')
    
    restore_var = tf.global_variables()
    ops = {
        'color_img':color_img,
        'pixel_label':pixel_label,
        'pixel_weight':pixel_weight,
        'coord_img':coord_img,
        'coord_label':coord_label,
        'coord_weight':coord_weight,
        'deeplab_logit':deeplab_logit,
        'deeplab_pred':deeplab_pred,
    }
    return ops, restore_var


def eval(sess, ops):
    dataset_recorder = eval_utils.get_recorder(NUM_CLASSES,   \
                                              'deeplab_2d')

    print("[INFO] Start Evaluation")
    num_scenes = len(VAL_DATASET_WHOLE_SCENE)
    for s in range(num_scenes):
        get_time = time.time()
        scene_data = VAL_DATASET_WHOLE_SCENE[s]
        get_time = time.time() - get_time
        scene_recorder = eval_utils.get_recorder(NUM_CLASSES,   \
                                                'deeplab_2d')

        num_view = scene_data['num_view'] 
        num_batch = math.ceil(float(num_view)/BATCH_SIZE)
        for bid in range(0, num_batch):
            start_vid = bid * BATCH_SIZE
            end_vid = start_vid + BATCH_SIZE 
            if end_vid > num_view:
                end_vid = num_view
            feed_time = time.time()
            feed_dict = {
                ops['color_img']: scene_data['color_img_list'][start_vid:end_vid],
                ops['pixel_label']: scene_data['pixel_label_list'][start_vid:end_vid], 
                ops['pixel_weight']: scene_data['pixel_weight_list'][start_vid:end_vid], 
                ops['coord_img']: scene_data['coord_img_list'][start_vid:end_vid],
                ops['coord_label']: scene_data['coord_label_list'][start_vid:end_vid],
                ops['coord_weight']: scene_data['coord_weight_list'][start_vid:end_vid]}

            # label_dir {path/to/dir}/scnene_name/label/frame_name.png""
            color_img_val, pixel_label_val, pixel_weight_val, \
            deeplab_pred_val, deeplab_logit_val = sess.run([ 
                ops['color_img'], 
                ops['pixel_label'], 
                ops['pixel_weight'], 
                ops['deeplab_pred'],
                ops['deeplab_logit']],
                feed_dict=feed_dict)
            feed_time = time.time()-feed_time

            # stretch 
            eval_time = time.time()
            pixel_weight_val = pixel_weight_val.reshape(-1)
            pixel_label_val = pixel_label_val.reshape(-1)
            deeplab_pred_val = deeplab_pred_val.reshape(-1)
            # scene record
            eval_utils.record(NUM_CLASSES, scene_recorder,        \
                   deeplab_2d_label = pixel_label_val * pixel_weight_val.astype(np.bool).astype(np.int32),\
                   deeplab_2d_pred = deeplab_pred_val)
            # dataset record
            eval_utils.record(NUM_CLASSES, dataset_recorder,        \
                   deeplab_2d_label = pixel_label_val * pixel_weight_val.astype(np.bool).astype(np.int32),\
                   deeplab_2d_pred = deeplab_pred_val)
            eval_time = time.time()-eval_time
            print('\r {}/{}, {} get time {}, {}/{}, feed time {}, eval time {}' \
                  .format(s, num_scenes, scene_data['scene_name'], get_time, end_vid, num_view, feed_time, eval_time), end='')

        eval_utils.evaluate_score(scene_recorder)
        print('\n----- {} score ----'.format(scene_data['scene_name']))
        eval_utils.log_score(scene_recorder, NUM_CLASSES, log_func=print)
    eval_utils.evaluate_score(dataset_recorder)
    log_string('\n----- Whole Dataset score ----')
    eval_utils.log_score(dataset_recorder, NUM_CLASSES, log_func=log_string, list_method=True)

    return dataset_recorder['deeplab_2d']['avg_iou']


def run():
    ops, restore_vars = build_graph()

    #Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    sess = tf.Session(config=config)
    saver = tf.train.Saver(var_list=restore_vars)

    best_miou = -1
    best_model = ''
    snapshots = utils.get_models(SNAPSHOT_DIR, MODEL_NAME) 
    snapshots_tested = []
    snapshots_to_test = list(set(snapshots) - set(snapshots_tested))

    while True:
        if len(snapshots_to_test) > 0:
            global_init = tf.global_variables_initializer()
            local_init = tf.local_variables_initializer()
            sess.run([global_init, local_init])
            snapshots_to_test = utils.human_sort(snapshots_to_test)
            model_to_test = snapshots_to_test[0]
            saver.restore(sess, model_to_test)
            log_string('\n[INFO] '+str(datetime.now())+', '+model_to_test+' has been restored')
            miou = eval(sess, ops)
            if miou > best_miou:
                best_miou = miou
                best_model = model_to_test
                log_string('\n\tmodel {} reaches current best Mean IoU: {}'.format(model_to_test, miou))
            else:
                log_string('\n\tmodel {} Mean IoU: {}'.format(model_to_test, miou))
            snapshots_tested.append(snapshots_to_test[0])
        else:
            log_string('\nsleep 15 minutes ...')
            time.sleep(60*15)
        snapshots = utils.get_models(SNAPSHOT_DIR, MODEL_NAME) 
        snapshots_to_test = list(set(snapshots) - set(snapshots_tested))

if __name__ == "__main__":
    run()

