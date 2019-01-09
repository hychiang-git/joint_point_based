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

# dataset info
NUM_CLASSES = 21
IGNORE_LABEL = 0
SCENE_POINTS = 16384
VOLUME_POINTS = 8192 
BATCH_SIZE = 4 
USE_FEATURE = 'syn_0.5'
DATA_PATH = '/tmp3/hychiang/scannetv2_data/'
VAL_DATASET_WHOLE_SCENE = scannet_dataset.ScannetDatasetVal(
    root = DATA_PATH, 
    num_classes=NUM_CLASSES,
    vpoints=VOLUME_POINTS,
    spoints=SCENE_POINTS,
    use_feature=USE_FEATURE,
    split='val')

# model snapshots dir
SNAPSHOT_DIR = os.path.join('log', 'train_log_{}'.format(USE_FEATURE), 'snapshots')
MODEL_NAME = 'model'

# logging
LOG_DIR = os.path.join('log', 'val_log_{}'.format(USE_FEATURE))
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
    coord_batch = tf.placeholder(dtype=tf.float32, shape=(None, VOLUME_POINTS, 265))
    label_batch = tf.placeholder(dtype=tf.int32, shape=(None, VOLUME_POINTS))
    weight_batch = tf.placeholder(dtype=tf.float32, shape=(None, VOLUME_POINTS))
    scene_point_batch = tf.placeholder(dtype=tf.float32, shape=(None, SCENE_POINTS, 265))
    # jointly encoder
    with tf.variable_scope('pointnet2') as scope:
        pointnet_logit, pointnet_end_point = \
            model.pointnet2(
                point_cloud=coord_batch,
                scene_point=scene_point_batch,
                is_training=is_training,
                bn_decay=None,
                num_class = NUM_CLASSES)
    with tf.variable_scope('pointnet2_loss'):
        pointnet_loss = model.get_loss(
            logits=pointnet_logit,
            labels=label_batch,
            num_classes=NUM_CLASSES,
            ignore_label=IGNORE_LABEL,
            weights=weight_batch)

    pointnet_pred = tf.argmax(pointnet_logit, axis=-1) 
    log_string('[INFO] PointNet2 Logit:\n'+str(pointnet_logit)+'\n--\n')
    log_string('[INFO] PointNet2 Prediction:\n'+str(pointnet_pred)+'\n--\n')
    
    restore_var = tf.global_variables()
    ops = {
           'coord_batch': coord_batch,
           'label_batch': label_batch,
           'weight_batch': weight_batch,
           'scene_point_batch': scene_point_batch,
           'pointnet_logit': pointnet_logit,
           'pointnet_pred': pointnet_pred,
           'pointnet_loss': pointnet_loss
    }
    return ops, restore_var


def eval(sess, ops):
    dataset_recorder = eval_utils.get_recorder(NUM_CLASSES,   \
                                              'pointnet_3d')
    print("[INFO] Start Evaluation")
    num_scenes = len(VAL_DATASET_WHOLE_SCENE)
    for s in range(num_scenes):
        scene_data = VAL_DATASET_WHOLE_SCENE[s]
        scene_recorder = eval_utils.get_recorder(NUM_CLASSES,   \
                                                'pointnet_3d')
        num_volume = scene_data['num_volume'] 
        num_batch = math.ceil(float(num_volume) / BATCH_SIZE)
        for bid in range(0, num_batch):
            start_vid = bid * BATCH_SIZE
            end_vid = bid * BATCH_SIZE + BATCH_SIZE
            if end_vid > num_volume:
                end_vid = num_volume
            print('\r{}/{}, {}, {}/{}'.format(s, num_scenes, scene_data['scene_name'], end_vid, num_volume), end='')
            start_time = time.time() 
            feed_dict = {
                ops['coord_batch']: scene_data['points'][start_vid:end_vid],
                ops['label_batch']: scene_data['labels'][start_vid:end_vid],
                ops['weight_batch']:scene_data['weights'][start_vid:end_vid],
                ops['scene_point_batch']:scene_data['scene_smpt'][start_vid:end_vid]
            }
            pointnet_loss_val, pointnet_logit_val, pointnet_pred_val = sess.run([
                ops['pointnet_loss'], 
                ops['pointnet_logit'],
                ops['pointnet_pred']
                ], feed_dict=feed_dict)
            # stretch 
            label_val = scene_data['labels'][start_vid:end_vid].reshape(-1)
            label_val = label_val * scene_data['masks'][start_vid:end_vid].reshape(-1)
            pointnet_pred_val = pointnet_pred_val.reshape(-1)
            # scene record
            eval_utils.record(NUM_CLASSES, scene_recorder,        \
                   pointnet_3d_label = label_val,\
                   pointnet_3d_pred = pointnet_pred_val)
            # dataset record
            eval_utils.record(NUM_CLASSES, dataset_recorder,        \
                   pointnet_3d_label = label_val,\
                   pointnet_3d_pred = pointnet_pred_val)

        eval_utils.evaluate_score(scene_recorder)
        print('\n----- {} score ----'.format(scene_data['scene_name']))
        eval_utils.log_score(scene_recorder, NUM_CLASSES, log_func=print)
    eval_utils.evaluate_score(dataset_recorder)
    log_string('\n----- Whole Dataset score ----')
    eval_utils.log_score(dataset_recorder, NUM_CLASSES, log_func=log_string, list_method=True)

    return dataset_recorder['pointnet_3d']['avg_iou']


def run():
    ops, restore_vars = build_graph()

    #Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    sess = tf.Session(config=config)
    saver = tf.train.Saver(var_list=restore_vars)

    pointnet_best_miou = -1
    pointnet_best_model = ''
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
            pointnet_miou = eval(sess, ops)
            if pointnet_miou > pointnet_best_miou:
                pointnet_best_miou = pointnet_miou
                pointnet_best_model = model_to_test
                log_string('\n\tmodel {} reaches current best Point Mean IoU: {}'.format(model_to_test, pointnet_miou))
            else:
                log_string('\n\tmodel {} Point Mean IoU: {}'.format(model_to_test, pointnet_miou))
            snapshots_tested.append(snapshots_to_test[0])
        else:
            log_string('\nsleep 15 minutes ...')
            time.sleep(60*15)
        snapshots = utils.get_models(SNAPSHOT_DIR, MODEL_NAME) 
        snapshots_to_test = list(set(snapshots) - set(snapshots_tested))

if __name__ == "__main__":
    run()

