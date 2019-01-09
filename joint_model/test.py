import sys
import os
import time
import math
import argparse
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

# params
parser = argparse.ArgumentParser()
parser.add_argument('--restore_model', required=True, help='path to testing model')
parser.add_argument('--split', required=True, help='train val test split')
parser.add_argument('--use_feature',default="feature", help='use what image features: feature/syn_1.0/syn_0.5/syn_0.3/syn_0.1')
parser.add_argument('--spoints', type=int, default=16384, help='scene points number')
parser.add_argument('--stride', type=float, default=0.3, help='scene stride size')
parser.add_argument('--padding', type=float, default=0.5, help='scene padding size')
parser.add_argument('--unmask', dest='unmask', action='store_true', help='not use mask to predict center only')
parser.add_argument('--from_scene', type=int, default=0, help='the start index of all scenes')
parser.add_argument('--to_scene', type=int, default=-1, help='the end index of all scenes')
parser.set_defaults(unmask=False)
opt = parser.parse_args()
TEST_MODEL = opt.restore_model
SPLIT = opt.split
USE_FEATURE = opt.use_feature
SCENE_POINTS = opt.spoints
STRIDE= opt.stride
PADDING= opt.padding
UNMASK= opt.unmask
print(opt)

# dataset info
NUM_CLASSES = 21
IGNORE_LABEL = 0
DATA_PATH = '/tmp3/hychiang/scannetv2_data/'
TEST_DATASET = scannet_dataset.ScannetDatasetTest(
    root = DATA_PATH, 
    num_classes=NUM_CLASSES,
    split=SPLIT,
    use_feature=USE_FEATURE,
    scene_stride=STRIDE,
    spoints=SCENE_POINTS,
    scene_padding=PADDING)

FROM_SCENE = opt.from_scene
TO_SCENE = opt.to_scene
num_scene = len(TEST_DATASET) 
if TO_SCENE > num_scene or TO_SCENE < FROM_SCENE:
    TO_SCENE = num_scene
assert TO_SCENE > FROM_SCENE and TO_SCENE <= num_scene
print('[INFO] from {} to {}'.format(FROM_SCENE, TO_SCENE))

# logging
LOG_DIR = os.path.join('log', 'test3d_log_{}'.format(USE_FEATURE), SPLIT, 'spts{}'.format(SCENE_POINTS))
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
PRED_DIR = os.path.join(LOG_DIR, 'pred_pad_'+str(PADDING)+'_stride_'+str(STRIDE)+'_unmask_'+str(UNMASK))
if not os.path.exists(PRED_DIR):
    os.mkdir(PRED_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_test.txt'), 'w')
# logging funcion
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def build_graph():
    is_training = False
    # placeholder 
    coord_batch = tf.placeholder(dtype=tf.float32, shape=(1, None, 265))
    scene_point_batch = tf.placeholder(dtype=tf.float32, shape=(1, SCENE_POINTS, 265))
    # jointly encoder
    with tf.variable_scope('pointnet2') as scope:
        pointnet_logit, pointnet_end_point = \
            model.pointnet2(
                point_cloud=coord_batch,
                scene_point=scene_point_batch,
                is_training=is_training,
                bn_decay=None,
                num_class = NUM_CLASSES)

    pointnet_pred = tf.argmax(pointnet_logit, axis=-1) 
    pointnet_logit = tf.nn.softmax(pointnet_logit, dim=-1)
    log_string('[INFO] PointNet2 Logit:\n'+str(pointnet_logit)+'\n--\n')
    log_string('[INFO] PointNet2 Prediction:\n'+str(pointnet_pred)+'\n--\n')
    
    restore_var = tf.global_variables()
    ops = {
           'coord_batch': coord_batch,
           'scene_point_batch': scene_point_batch,
           'pointnet_logit': pointnet_logit,
           'pointnet_pred': pointnet_pred,
    }
    return ops, restore_var


def test(sess, ops):
    print("[INFO] Start Testing")
    num_scenes = len(TEST_DATASET)
    for s in range(FROM_SCENE, TO_SCENE):
        scene_data = TEST_DATASET[s]
        scene_recorder = eval_utils.get_recorder(NUM_CLASSES,   \
                                                'pointnet_3d')
        num_volume = scene_data['num_volume'] 
        scene_logit = np.zeros((len(scene_data['scene_point']), NUM_CLASSES), dtype=np.float32)
        for vid in range(0, num_volume):
            print('\r[{}:{}:{}], {}, {}/{}'.format(FROM_SCENE, s, TO_SCENE, scene_data['scene_name'], vid, num_volume), end='')
            start_time = time.time() 
            feed_dict = {
                ops['coord_batch']: scene_data['point_list'][vid],
                ops['scene_point_batch']:scene_data['scene_list'][vid]
            }
            pointnet_logit_val, pointnet_pred_val = sess.run([
                ops['pointnet_logit'],
                ops['pointnet_pred']
                ], feed_dict=feed_dict)
            # stretch 
            pointnet_logit_val = np.squeeze(pointnet_logit_val) # N x NC
            point_idx_val = np.squeeze(scene_data['pidx_list'][vid]) # N x NC
            mask_idx_val = np.squeeze(scene_data['mask_list'][vid]) # N x NC
            if UNMASK:
                scene_logit[point_idx_val, ...] += pointnet_logit_val
            else:
                valid_point = point_idx_val[mask_idx_val]
                valid_logit = pointnet_logit_val[mask_idx_val, ...]
                scene_logit[valid_point, ...] += valid_logit
        scene_pred = np.argmax(scene_logit, axis=-1)
        scene_pred_file = os.path.join(PRED_DIR, scene_data['scene_name']+'.txt')
        with open(scene_pred_file, 'w') as f:
            scene_pred = scene_pred.astype(np.int32)
            for v in range(len(scene_pred)):
                pred_eval_id = g_label_ids[scene_pred[v]]
                print(pred_eval_id, file=f)


def run():
    ops, restore_vars = build_graph()
    #Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver(var_list=restore_vars)
    global_init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()
    sess.run([global_init, local_init])
    saver.restore(sess, TEST_MODEL)
    log_string('\n[INFO] '+str(datetime.now())+', '+TEST_MODEL+' has been restored')
    test(sess, ops)

if __name__ == "__main__":
    run()

