import os
import sys
import time
import socket
import argparse
import numpy as np
import tensorflow as tf
import multiprocessing as mp
from PIL import Image
from datetime import datetime
from functools import partial

from core import model
from core import train_utils
from core import scannet_dataset

slim = tf.contrib.slim

# Data path and parameters
NUM_CLASSES = 21
IGNORE_LABEL = 0
SNAPSHOT_DIR = './snapshots/'

# restore model
DEEPLAB_INIT_MODEL = './init_weights/deeplabv3_xception_init'
#DEEPLAB_RESTORE_MODEL = './init_weights/deeplab_model_235.ckpt' 
DEEPLAB_RESTORE_MODEL = None
POINTNET_RESTORE_MODEL = None
GLOBAL_RESTORE_MODEL = None

# data path
IMAGE_SIZE = (480, 640)
COORD_SIZE = (120, 160)
DATA_PATH = os.path.join('/tmp3/hychiang/data/')
TRAIN_DATASET = scannet_dataset.ScannetDatasetTrain(
    root = DATA_PATH, 
    get_depth=False,
    get_coord=False,
    point_from_depth=False,
    num_classes=NUM_CLASSES)

# Training configuration 
parser = argparse.ArgumentParser()
parser.add_argument('--save_freq', type=int, default=5, help='Save model frequency (every n epoches) [default: 5]')
parser.add_argument('--max_epoch', type=int, default=501, help='Epoch to run [default: 501]')
parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPU [default: 1]')
parser.add_argument('--gpu_bsize', type=int, default=4, help='Batch size in a GPU [default: 4]')
parser.add_argument('--slow_start_step', type=int, default=0, help='Smaller learning rate for before slow_start_step [default: 0]')

FLAGS = parser.parse_args()
SAVE_FREQ = FLAGS.save_freq
MAX_EPOCH = FLAGS.max_epoch
NUM_GPU = FLAGS.num_gpus 
GPU_BATCH_SIZE = FLAGS.gpu_bsize 
BATCH_SIZE = NUM_GPU * GPU_BATCH_SIZE 
TRAIN_STEPS = MAX_EPOCH * int( len(TRAIN_DATASET) / BATCH_SIZE)
SLOW_START_STEP = FLAGS.slow_start_step

# Logging
LOG_DIR = os.path.join('log', 'train_log')
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
LOG_TRAIN_DIR = os.path.join(LOG_DIR, 'train')
if not os.path.exists(LOG_TRAIN_DIR): os.mkdir(LOG_TRAIN_DIR)
LOG_SNAPSHOT_DIR = os.path.join(LOG_DIR, 'snapshots')
if not os.path.exists(LOG_SNAPSHOT_DIR): os.mkdir(LOG_SNAPSHOT_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')
print(str(FLAGS))

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

# other global vars
HOSTNAME = socket.gethostname()
POOL = None

def build_train_graph():

    # placeholder 
    color_img_batch = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    pixel_label_batch = tf.placeholder(dtype=tf.int32, shape=(BATCH_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1], 1))
    pixel_weight_batch = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1], 1))
    coord_img_batch = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1], 6))
    coord_label_batch = tf.placeholder(dtype=tf.int32, shape=(BATCH_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1], 1))
    coord_weight_batch = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1], 1))
    global_step_pl = tf.placeholder(dtype=tf.float32, shape=())
    # model training setup
    is_training = True 
    learning_rate = train_utils.model_learning_rate(
        global_step=global_step_pl,
        learning_policy='poly',
        base_learning_rate=0.0001,
        learning_rate_decay_step=2000, 
        learning_rate_decay_factor=0.1,
        training_number_of_steps=TRAIN_STEPS, 
        learning_power=0.9,
        slow_start_step=SLOW_START_STEP,
        slow_start_learning_rate=0.00001)
    bn_decay = train_utils.get_bn_decay(
        global_step=global_step_pl,
        bn_init_decay=0.5,
        bn_decay_step=2000,
        bn_decay_rate=0.5,
        bn_decay_clip=0.99)
    deeplab_learning_rate = learning_rate
    deeplab_optimizer = tf.train.MomentumOptimizer(deeplab_learning_rate, 0.9)

    deeplab_grads, deeplab_logits, deeplab_losses = list(), list(), list()
    with tf.variable_scope(tf.get_variable_scope()):
        # build deeplab at gpu %d
        for i in range(NUM_GPU):
            gpu_batch_start = i * GPU_BATCH_SIZE
            gpu_batch_end = gpu_batch_start + GPU_BATCH_SIZE
            with tf.name_scope('joint_encoder') as scope:
                feature, deeplab_end_point = \
                    model.joint_encoder(
                        images=color_img_batch[gpu_batch_start:gpu_batch_end],
                        coords=coord_img_batch[gpu_batch_start:gpu_batch_end, :, :, 0:3],
                        batch_size = BATCH_SIZE,
                        output_stride=16,
                        depth_init_size=COORD_SIZE,
                        bn_decay=bn_decay,
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
                    tf.shape(pixel_label_batch)[1:3], 
                    align_corners=True)
                deeplab_loss = model.get_loss(
                    logits=deeplab_logit,
                    labels=pixel_label_batch[gpu_batch_start:gpu_batch_end],
                    num_classes=NUM_CLASSES,
                    ignore_label=IGNORE_LABEL,
                    weights=pixel_weight_batch[gpu_batch_start:gpu_batch_end])
            # gradient 
            with tf.variable_scope('deeplab') as scope:
                # Get deeplab loss, gradient, variables and gradient multiplier
                last_layers = model.extra_layer_scopes()
                deeplab_gv_all = deeplab_optimizer.compute_gradients(deeplab_loss)
                deeplab_gv = [gv for gv in deeplab_gv_all if 'pointnet' not in gv[1].name]
                deeplab_grad_mult = train_utils.model_gradient_multipliers(last_layers, 1.0, deeplab_gv)
                deeplab_gv = train_utils.multiply_gradients(deeplab_gv, deeplab_grad_mult)
                deeplab_grads.append(deeplab_gv)
                deeplab_logits.append(deeplab_logit)
                deeplab_losses.append(deeplab_loss)
            print('[INFO] Build Network at GPU:', i)
            # resue variable for next gpu
            tf.get_variable_scope().reuse_variables()
    # deeplab grad and vars
    deeplab_restore_vars, deeplab_save_vars = train_utils.model_restore_save_vars('deeplab')
    deeplab_final_logits = tf.concat(deeplab_logits, axis=0)
    deeplab_avg_loss = tf.reduce_mean(deeplab_losses, name='deeplab_loss')
    deeplab_avg_grads_vars = train_utils.average_gradients(deeplab_grads)
    # train op: apply gradient
    deeplab_train_op = deeplab_optimizer.apply_gradients(deeplab_avg_grads_vars)

    #global_vars = deeplab_save_vars + pointnet_save_vars
    global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    ops = {'global_step': global_step_pl,
           'color_img_batch': color_img_batch,
           'pixel_label_batch': pixel_label_batch,
           'pixel_weight_batch': pixel_weight_batch,
           'coord_img_batch': coord_img_batch,
           'coord_label_batch': coord_label_batch,
           'coord_weight_batch': coord_weight_batch,
           'deeplab_logit': deeplab_final_logits,
           'deeplab_loss': deeplab_avg_loss,
           'deeplab_train_op': deeplab_train_op,
           'deeplab_save_vars': deeplab_save_vars,
           'deeplab_restore_vars': deeplab_restore_vars,
           'global_vars': global_vars}
    return ops


def get_sample(sid, dataset, idxs, start_idx):
    scene_name, color_img, depth_img, pixel_label, pixel_weight, coord_img, coord_label, coord_weight, valid_coord_idx = dataset[idxs[sid+start_idx]]
    return color_img, depth_img, pixel_label, pixel_weight, coord_img, coord_label, coord_weight 

def get_batch_wdp(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    get_sample_partial =partial(get_sample, dataset=dataset, idxs=idxs, start_idx=start_idx)
    res = POOL.map(get_sample_partial, range(bsize))
    color_img_batch = np.array([r[0] for r in res])
    depth_img_batch = np.array([r[1] for r in res])
    pixel_label_batch = np.array([r[2] for r in res])
    pixel_weight_batch = np.array([r[3] for r in res])
    coord_img_batch = np.array([r[4] for r in res])
    coord_label_batch = np.array([r[5] for r in res])
    coord_weight_batch = np.array([r[6] for r in res])
    return color_img_batch, depth_img_batch, pixel_label_batch, pixel_weight_batch, \
           coord_img_batch, coord_label_batch, coord_weight_batch

def train_one_epoch(epoch, sess, ops):
    """ ops: dict mapping from string to tf ops """
    # Shuffle train samples
    train_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)
    num_batches = len(TRAIN_DATASET)/BATCH_SIZE
    
    log_string(str(datetime.now()))

    deeplab_total_correct, deeplab_total_seen, deeplab_loss_sum = 0, 0, 0
    avg_btime, avg_ftime = 0, 0
    for batch_idx in range(int(num_batches)):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        
        start_time = time.time() 
        color_img_batch, depth_img_batch, pixel_label_batch, pixel_weight_batch, \
        coord_img_batch, coord_label_batch, coord_weight_batch \
             = get_batch_wdp(TRAIN_DATASET, train_idxs, start_idx, end_idx)
        avg_btime += time.time()-start_time
        # Augment batched point clouds by rotation
        start_time = time.time() 
        feed_dict = {
            ops['global_step']: epoch*num_batches + batch_idx,
            ops['color_img_batch']: color_img_batch,
            ops['pixel_label_batch']: pixel_label_batch,
            ops['pixel_weight_batch']: pixel_weight_batch,
            ops['coord_img_batch']: coord_img_batch,
            ops['coord_label_batch']: coord_label_batch,
            ops['coord_weight_batch']: coord_weight_batch}
        _, step, deeplab_loss_val, deeplab_logit_val = sess.run([
            ops['deeplab_train_op'], 
            ops['global_step'],
            ops['deeplab_loss'], 
            ops['deeplab_logit'],
            ], feed_dict=feed_dict)
        pixel_label_batch = np.squeeze(pixel_label_batch)
        pixel_weight_batch = np.squeeze(pixel_weight_batch)

        deeplab_pred_val = np.squeeze(np.argmax(deeplab_logit_val, 3))
        deeplab_correct = np.sum(deeplab_pred_val == pixel_label_batch)
        deeplab_total_correct += deeplab_correct
        deeplab_total_seen += np.sum(pixel_label_batch.astype(np.bool)) 
        deeplab_loss_sum += deeplab_loss_val

        avg_ftime += time.time()-start_time
        ## Augment batched point clouds by rotation
        if (batch_idx+1)%10 == 0:
            log_string(' -- %03d / %03d --' % (batch_idx+1, num_batches))
            log_string('average get batch time: %f, average forwarding time: %f' % (avg_btime/ 10, avg_ftime/10))
            log_string('deeplab loss: %f' % (deeplab_loss_sum / 10))
            log_string('deeplab accuracy: %f' % (deeplab_total_correct / float(deeplab_total_seen)))
            deeplab_total_correct, deeplab_total_seen, deeplab_loss_sum = 0, 0, 0
            avg_btime, avg_ftime = 0, 0


def train():
    # build graph
    ops = build_train_graph()
    #Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement=True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)

    if DEEPLAB_INIT_MODEL !=None:
        deeplab_restore_vars = train_utils.restore_vars(DEEPLAB_INIT_MODEL, 'deeplab')
        deeplab_loader = tf.train.Saver(var_list=deeplab_restore_vars, max_to_keep=10)    # Restore except logit layer
        deeplab_loader.restore(sess, DEEPLAB_INIT_MODEL)
        print ("[INFO] Deeplab Initial Model ", DEEPLAB_INIT_MODEL, ' has been restored except logits layer')
    if DEEPLAB_RESTORE_MODEL !=None:
        deeplab_restore_vars = train_utils.restore_vars(DEEPLAB_RESTORE_MODEL, 'deeplab')
        deeplab_loader = tf.train.Saver(var_list=deeplab_restore_vars, max_to_keep=10)  
        deeplab_loader.restore(sess, DEEPLAB_RESTORE_MODEL)
        print ("[INFO] Deeplab  Model ", DEEPLAB_RESTORE_MODEL, ' has been restored.')

    global_saver = tf.train.Saver(var_list=ops['global_vars'], max_to_keep=None)    # restore global model
    if GLOBAL_RESTORE_MODEL !=None:
        global_saver.restore(sess, GLOBAL_RESTORE_MODEL)
        print ("[INFO] Global Model ", GLOBAL_RESTORE_MODEL, ' has been restored')
    
    print ("[INFO] Start Training")
    for epoch in range(MAX_EPOCH):
        log_string('**** EPOCH %03d ****' % (epoch))
        sys.stdout.flush()
        train_one_epoch(epoch, sess, ops)
        if epoch % SAVE_FREQ == 0:
            save_path = global_saver.save(sess, os.path.join(LOG_SNAPSHOT_DIR, "model_%03d.ckpt"%(epoch)))
            log_string("GLOBAL Model saved in file: %s" % (save_path))

if __name__ == "__main__":
    POOL = mp.Pool(BATCH_SIZE)
    train()
