import sys
import os
import time
import math
import imageio
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
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
parser.add_argument('--save_image', dest='save_image', action='store_true')
parser.add_argument('--data_dir', type=str, default='/tmp3/hychiang/scannetv2_data/', help='data directory')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
parser.add_argument('--from_scene', type=int, default=0, help='the start index of all scenes')
parser.add_argument('--to_scene', type=int, default=-1, help='the end index of all scenes')
parser.set_defaults(save_feature=False, save_image=False)
opt = parser.parse_args()

SAVE_IMAGE = opt.save_image
TEST_MODEL = opt.restore_model
BATCH_SIZE = opt.batch_size 
SPLIT = opt.split
DATA_PATH = opt.data_dir
print(opt)

NUM_CLASSES = 21
IGNORE_LABEL = 0
IMAGE_SIZE = (480, 640)
COORD_SIZE = (120, 160)
VAL_DATASET_WHOLE_SCENE = scannet_dataset.ScannetDatasetImageTest( \
        root=DATA_PATH,
        num_classes=NUM_CLASSES,
        split=SPLIT,
        frame_skip=1)

FROM_SCENE = opt.from_scene
TO_SCENE = opt.to_scene
num_scene = len(VAL_DATASET_WHOLE_SCENE) 
if TO_SCENE > num_scene or TO_SCENE==-1:
    TO_SCENE = num_scene
assert TO_SCENE > FROM_SCENE
print('[INFO] from {} to {}'.format(FROM_SCENE, TO_SCENE))

# logging
LOG_DIR = os.path.join('log', 'test2d_log', SPLIT)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
PRED_DIR = os.path.join(LOG_DIR, 'pred')
if not os.path.exists(PRED_DIR):
    os.makedirs(PRED_DIR)
PRED_VIS_DIR = os.path.join(LOG_DIR, 'pred_vis')
if not os.path.exists(PRED_VIS_DIR):
    os.makedirs(PRED_VIS_DIR)
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

def visualize_label_image(filename, image):
    height = image.shape[0]
    width = image.shape[1]
    vis_image = np.zeros([height, width, 3], dtype=np.uint8)
    color_palette = scannet_util.create_color_palette()
    for idx, color in enumerate(color_palette):
        vis_image[image==idx] = color
    imageio.imwrite(filename, vis_image)

def test(sess, ops, save_image=False):
    print("[INFO] Start Evaluation")
    for s in range(FROM_SCENE, TO_SCENE):
        test_time = time.time()
        get_time = time.time()
        scene_data = VAL_DATASET_WHOLE_SCENE[s]
        get_time = time.time() - get_time
        print('[{} : {} : {}] {}, get time {}' \
              .format(FROM_SCENE, s, TO_SCENE, scene_data['scene_name'], get_time))

        scene_pred_dir = os.path.join(PRED_DIR, scene_data['scene_name'])
        if not os.path.exists(scene_pred_dir):
            os.makedirs(scene_pred_dir)
        if save_image is True:
            scene_pred_vis_dir = os.path.join(PRED_VIS_DIR, scene_data['scene_name'])
            if not os.path.exists(scene_pred_vis_dir):
                os.makedirs(scene_pred_vis_dir)
        num_view = scene_data['num_view'] 
        num_batch = math.ceil(float(num_view)/BATCH_SIZE)
        for bid in range(0, num_batch):
            feed_time = time.time()
            start_vid = bid * BATCH_SIZE
            end_vid = start_vid + BATCH_SIZE 
            if end_vid > num_view:
                end_vid = num_view
            num_samples = end_vid - start_vid

            feed_dict = {
                ops['color_img']: scene_data['color_img_list'][start_vid:end_vid]
            }

            color_img_val, deeplab_feature_val, deeplab_pred_val, deeplab_logit_val = sess.run([ 
                ops['color_img'], 
                ops['deeplab_feature'],
                ops['deeplab_pred'],
                ops['deeplab_logit']],
                feed_dict=feed_dict)

            pred_val_1D = deeplab_pred_val.reshape((num_samples, -1))
            # stretch 
            feed_time = time.time()-feed_time
            print('\r[{} : {} : {}] feed time {}, {}/{}' \
                  .format(FROM_SCENE, s, TO_SCENE, feed_time, end_vid, num_view), end='')
            for i in range(num_samples):
                image_height = deeplab_pred_val[i].shape[0]
                image_width = deeplab_pred_val[i].shape[1]
                # output pred scene.txt
                pred_img = Image.fromarray(deeplab_pred_val[i].astype(np.uint8))
                pred_png = os.path.join(scene_pred_dir, '%06d.png'%(bid*BATCH_SIZE+i) )
                pred_img.save(pred_png)
                if save_image is True:
                    pred_vis_png = os.path.join(scene_pred_vis_dir, '%06d.png'%(bid*BATCH_SIZE+i))
                    pred_vis = np.zeros(pred_val_1D[i].shape, dtype=np.uint8)
                    for l in range(len(pred_val_1D[i])):
                        pred_vis[l] = g_label_ids[pred_val_1D[i][l]]
                    pred_vis = pred_vis.reshape(image_height, image_width)
                    visualize_label_image(pred_vis_png, pred_vis)
        test_time = time.time() - test_time
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
    test(sess, ops, SAVE_IMAGE)

if __name__ == "__main__":
    run()

