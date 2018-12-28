import sys
import os
import time
import math
import argparse
import numpy as np
import tensorflow as tf
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
parser.add_argument('--save_feature', dest='save_feature', action='store_true')
parser.add_argument('--data_dir', type=str, default='/tmp3/hychiang/scannetv2_data/', help='data directory')
parser.add_argument('--use_image', type=float, default=1.0, help='percentage of images to back project')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
parser.add_argument('--from_scene', type=int, default=0, help='the start index of all scenes')
parser.add_argument('--to_scene', type=int, default=-1, help='the end index of all scenes')
parser.set_defaults(save_feature=False)
opt = parser.parse_args()

SAVE_FEATURE = opt.save_feature 
TEST_MODEL = opt.restore_model
BATCH_SIZE = opt.batch_size 
SPLIT = opt.split
DATA_PATH = opt.data_dir
USE_IMAGE_PERCENT = opt.use_image
print(opt)

FEATURE_DIMS = 256
NUM_CLASSES = 21
IGNORE_LABEL = 0
IMAGE_SIZE = (480, 640)
COORD_SIZE = (120, 160)
TEST_DATASET = scannet_dataset.ScannetDatasetTest( \
        root=DATA_PATH,
        num_classes=NUM_CLASSES,
        split=SPLIT,
        get_pixmeta=True,
        get_scene_point=True,
        use_image=USE_IMAGE_PERCENT)

FROM_SCENE = opt.from_scene
TO_SCENE = opt.to_scene
num_scene = len(TEST_DATASET) 
if TO_SCENE > num_scene or TO_SCENE==-1:
    TO_SCENE = num_scene
assert TO_SCENE > FROM_SCENE
print('[INFO] from {} to {}'.format(FROM_SCENE, TO_SCENE))

# logging
LOG_DIR = os.path.join('log', 'test3d_log', SPLIT+'_'+str(USE_IMAGE_PERCENT))
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
PRED_DIR = os.path.join(LOG_DIR, 'pred')
if not os.path.exists(PRED_DIR):
    os.mkdir(PRED_DIR)
FEAT_DIR = os.path.join(LOG_DIR, 'features')
if not os.path.exists(FEAT_DIR):
    os.mkdir(FEAT_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_test.txt'), 'w')
# logging funcion
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def build_graph():
    is_training = False
    # placeholder 
    color_img = tf.placeholder(dtype=tf.float32, shape=(None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    # jointly encoder
    with tf.name_scope('joint_encoder') as scope:
        feature, deeplab_end_point = \
            model.joint_encoder(
                images=color_img,
                batch_size = 1,
                output_stride=16,
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
        deeplab_logit = tf.image.resize_bilinear(
            deeplab_logit, 
            IMAGE_SIZE, 
            align_corners=True)

    deeplab_pred = tf.argmax(deeplab_logit, axis=-1) 
    deeplab_logit = tf.nn.softmax(deeplab_logit, dim=-1)
    deeplab_feature = tf.image.resize_bilinear(
        feature, 
        IMAGE_SIZE, 
        align_corners=True)
    deeplab_feature = tf.nn.l2_normalize(deeplab_feature, dim=-1)
    log_string('[INFO] Deeplab Feature:\n'+str(deeplab_feature)+'\n--\n')
    log_string('[INFO] Deeplab Logit:\n'+str(deeplab_logit)+'\n--\n')
    log_string('[INFO] Deeplab Prediction:\n'+str(deeplab_pred)+'\n--\n')
    
    restore_var = tf.global_variables()
    ops = {
        'color_img':color_img,
        'deeplab_feature':deeplab_feature,
        'deeplab_logit':deeplab_logit,
        'deeplab_pred':deeplab_pred,
    }
    return ops, restore_var


def test(sess, ops, save_feature=False):
    print("[INFO] Start Evaluation")
    for s in range(FROM_SCENE, TO_SCENE):
        test_time = time.time()
        get_time = time.time()
        scene_data = TEST_DATASET[s]
        get_time = time.time() - get_time
        print('[{} : {} : {}] {}, get time {}' \
              .format(FROM_SCENE, s, TO_SCENE, scene_data['scene_name'], get_time))

        scene_point = scene_data['scene_point'] 
        scene_logit = np.zeros((len(scene_point), NUM_CLASSES), dtype=np.float32)
        scene_feature = np.zeros((len(scene_point), FEATURE_DIMS), dtype=np.float32)
        num_view = scene_data['num_view'] 
        num_batch = math.ceil(float(num_view)/BATCH_SIZE)
        for bid in range(0, num_batch):
            feed_time = time.time()
            start_vid = bid * BATCH_SIZE
            end_vid = start_vid + BATCH_SIZE 
            if end_vid > num_view:
                end_vid = num_view
            feed_dict = {
                ops['color_img']: scene_data['color_img_list'][start_vid:end_vid],
            }

            color_img_val, deeplab_feature_val, deeplab_pred_val, deeplab_logit_val = sess.run([ 
                ops['color_img'], 
                ops['deeplab_feature'],
                ops['deeplab_pred'],
                ops['deeplab_logit']],
                feed_dict=feed_dict)

            # stretch 
            deeplab_pred_val = deeplab_pred_val.reshape(-1)
            deeplab_logit_val = deeplab_logit_val.reshape((-1, NUM_CLASSES))
            deeplab_feature_val = deeplab_feature_val.reshape((-1, FEATURE_DIMS))
            mesh_vertex, v1w, v2w, v3w = eval_utils.barycentric_weight(np.array(scene_data['pixel_meta_list'][start_vid:end_vid]))
            scene_logit[mesh_vertex[:,0]] += v1w*deeplab_logit_val
            scene_logit[mesh_vertex[:,1]] += v2w*deeplab_logit_val
            scene_logit[mesh_vertex[:,2]] += v3w*deeplab_logit_val
            if save_feature is True:
                scene_feature[mesh_vertex[:,0]] += v1w*deeplab_feature_val
                scene_feature[mesh_vertex[:,1]] += v2w*deeplab_feature_val
                scene_feature[mesh_vertex[:,2]] += v3w*deeplab_feature_val
            feed_time = time.time()-feed_time
            print('\r[{} : {} : {}] feed time {}, {}/{}' \
                  .format(FROM_SCENE, s, TO_SCENE, feed_time, end_vid, num_view), end='')

        # output pred scene.txt
        output_time = time.time()
        scene_pred = np.argmax(scene_logit, axis=-1)
        scene_pred_file = os.path.join(PRED_DIR, scene_data['scene_name']+'.txt')
        with open(scene_pred_file, 'w') as f:
            scene_pred = scene_pred.astype(np.int32)
            for v in range(len(scene_pred)):
                pred_eval_id = g_label_ids[scene_pred[v]]
                print(pred_eval_id, file=f)
        if save_feature is True:
            scene_feature = scene_feature / (np.sqrt(np.sum(scene_feature**2, axis=-1, keepdims=True)) + 1e-8)
            scene_feature_file = os.path.join(FEAT_DIR, scene_data['scene_name'])
            np.save(scene_feature_file, scene_feature)
            output_time = time.time() - output_time
            test_time = time.time() - test_time
            print('\n[{} : {} : {}] {}, {}, {}, {}, output time {}' \
                  .format(FROM_SCENE, s, TO_SCENE, scene_pred_file, scene_feature_file, \
                   scene_feature.shape, scene_point.shape, output_time))
        else:
            output_time = time.time() - output_time
            test_time = time.time() - test_time
            print('\n[{} : {} : {}] {}, {}, output time {}' \
                  .format(FROM_SCENE, s, TO_SCENE, scene_pred_file, scene_point.shape, output_time))
        print('[{} : {} : {}] test time {}' \
              .format(FROM_SCENE, s, TO_SCENE, test_time))
        print('-------------------------')

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
    test(sess, ops, SAVE_FEATURE)

if __name__ == "__main__":
    run()

