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

# restore model
POINTNET_RESTORE_MODEL = None
GLOBAL_RESTORE_MODEL = None

# Data path and parameters
NUM_CLASSES = 21
IGNORE_LABEL = 0
SCENE_POINTS = 16384
VOLUME_POINTS = 8192 
USE_FEATURE = 'syn_0.5'
DATA_PATH = '/tmp3/hychiang/scannetv2_data/'
TRAIN_DATASET = scannet_dataset.ScannetDatasetTrain(
                    root=DATA_PATH,
                    num_classes=NUM_CLASSES,
                    vpoints=VOLUME_POINTS,
                    spoints=SCENE_POINTS,
                    split='train',
                    use_feature=USE_FEATURE,
                    dropout=True,
                    aug_z=True
                    )

# Training configuration 
parser = argparse.ArgumentParser()
parser.add_argument('--save_freq', type=int, default=5, help='Save model frequency (every n epoches) [default: 5]')
parser.add_argument('--max_epoch', type=int, default=501, help='Epoch to run [default: 501]')
parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPU [default: 1]')
parser.add_argument('--gpu_bsize', type=int, default=6, help='Batch size in a GPU [default: 6]')
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
LOG_DIR = os.path.join('log', 'train_log_{}'.format(USE_FEATURE))
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
    coord_batch = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, VOLUME_POINTS, 265))
    label_batch = tf.placeholder(dtype=tf.int32, shape=(BATCH_SIZE, VOLUME_POINTS))
    weight_batch = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, VOLUME_POINTS))
    scene_point_batch = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, SCENE_POINTS, 265))
    global_step_pl = tf.placeholder(dtype=tf.float32, shape=())
    is_training = True 
    learning_rate = train_utils.model_learning_rate(
        global_step=global_step_pl,
        learning_policy='poly',
        base_learning_rate=0.001,
        learning_rate_decay_step=2000, 
        learning_rate_decay_factor=0.1,
        training_number_of_steps=TRAIN_STEPS, 
        learning_power=0.9,
        slow_start_step=SLOW_START_STEP,
        slow_start_learning_rate=0.0001)
    bn_decay = train_utils.get_bn_decay(
        global_step=global_step_pl,
        bn_init_decay=0.5,
        bn_decay_step=2000,
        bn_decay_rate=0.5,
        bn_decay_clip=0.99)
    #pointnet_optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    pointnet_optimizer = tf.train.AdamOptimizer(learning_rate)

    pointnet_logits, pointnet_grads, pointnet_losses = list(), list(), list()
    with tf.variable_scope(tf.get_variable_scope()):
        # build deeplab at gpu %d
        for i in range(NUM_GPU):
            with tf.device('/gpu:%d' % i):
                gpu_batch_start = i*GPU_BATCH_SIZE
                gpu_batch_end = gpu_batch_start + GPU_BATCH_SIZE
                with tf.variable_scope('pointnet2') as scope:
                    pointnet_logit, pointnet_end_point = \
                        model.pointnet2(
                            point_cloud=coord_batch[gpu_batch_start:gpu_batch_end],
                            scene_point=scene_point_batch[gpu_batch_start:gpu_batch_end],
                            is_training=is_training,
                            bn_decay=bn_decay,
                            num_class = NUM_CLASSES)
                with tf.variable_scope('pointnet2_loss'):
                    pointnet_loss = model.get_loss(
                        logits=pointnet_logit,
                        labels=label_batch[gpu_batch_start:gpu_batch_end],
                        num_classes=NUM_CLASSES,
                        ignore_label=IGNORE_LABEL,
                        weights=weight_batch[gpu_batch_start:gpu_batch_end])
                print('gpu:', i, ', strart: ', gpu_batch_start, ', end:', gpu_batch_end)

                # gradient 
                with tf.variable_scope('pointnet2') as scope:
                    # Get pointnet loss, gradient and variables
                    pointnet_gv_all = pointnet_optimizer.compute_gradients(pointnet_loss)
                    pointnet_gv = [gv for gv in pointnet_gv_all if 'deeplab' not in gv[1].name]
                    grad_mult = train_utils.model_gradient_multipliers(['logits'], 1.0, pointnet_gv) 
                    pointnet_gv = train_utils.multiply_gradients(pointnet_gv, grad_mult)
                    pointnet_grads.append(pointnet_gv)
                    pointnet_logits.append(pointnet_logit)
                    pointnet_losses.append(pointnet_loss)
            print('[INFO] Build Network at GPU:', i)
            # resue variable for next gpu
            tf.get_variable_scope().reuse_variables()
    # pointnet2 grad and vars
    pointnet_restore_vars, pointnet_save_vars = train_utils.model_restore_save_vars('pointnet2')
    pointnet_final_logits = tf.concat(pointnet_logits, axis=0)
    pointnet_avg_loss = tf.reduce_mean(pointnet_losses, name='avg_loss')
    pointnet_avg_grads_vars = train_utils.average_gradients(pointnet_grads)
    # train op: apply gradient
    pointnet_train_op = pointnet_optimizer.apply_gradients(pointnet_avg_grads_vars)

    global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    ops = {'global_step': global_step_pl,
           'coord_batch': coord_batch,
           'label_batch': label_batch,
           'weight_batch': weight_batch,
           'scene_point_batch': scene_point_batch,
           'pointnet_logit': pointnet_final_logits,
           'pointnet_loss': pointnet_avg_loss,
           'pointnet_train_op': pointnet_train_op,
           'pointnet_save_vars': pointnet_save_vars,
           'pointnet_restore_vars': pointnet_restore_vars,
           'global_vars': global_vars}
    return ops


def get_sample(sid, dataset, idxs, start_idx):
    scene_name, scene_point, points, label, weight, scene_smpt = dataset[idxs[sid+start_idx]]
    return points, label, weight, scene_smpt

def get_batch_wdp(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    get_sample_partial =partial(get_sample, dataset=dataset, idxs=idxs, start_idx=start_idx)
    res = POOL.map(get_sample_partial, range(bsize))
    coord_batch = np.array([r[0] for r in res])
    label_batch = np.array([r[1] for r in res])
    weight_batch = np.array([r[2] for r in res])
    scene_smpt_batch = np.array([r[3] for r in res])
    return coord_batch, label_batch, weight_batch, scene_smpt_batch

def train_one_epoch(epoch, sess, ops):
    """ ops: dict mapping from string to tf ops """
    # Shuffle train samples
    train_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)
    num_batches = len(TRAIN_DATASET)/BATCH_SIZE
    
    log_string(str(datetime.now()))

    pointnet_total_correct, pointnet_total_seen, pointnet_loss_sum = 0, 0, 0
    avg_btime, avg_ftime = 0, 0
    for batch_idx in range(int(num_batches)):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        
        start_time = time.time() 
        coord_batch, label_batch, weight_batch, scene_smpt_batch \
        = get_batch_wdp(TRAIN_DATASET, train_idxs, start_idx, end_idx)
        avg_btime += time.time()-start_time
        ## Augment batched point clouds by rotation
        start_time = time.time() 
        feed_dict = {
            ops['global_step']: epoch*num_batches + batch_idx,
            ops['coord_batch']: coord_batch,
            ops['label_batch']: label_batch,
            ops['weight_batch']:weight_batch,
            ops['scene_point_batch']:scene_smpt_batch
        }
        _, step, pointnet_loss_val, pointnet_logit_val = sess.run([
            ops['pointnet_train_op'], 
            ops['global_step'],
            ops['pointnet_loss'], 
            ops['pointnet_logit']
            ], feed_dict=feed_dict)
        label_batch = np.squeeze(label_batch)
        weight_batch = np.squeeze(weight_batch)

        pointnet_pred_val = np.squeeze(np.argmax(pointnet_logit_val, 2))
        pointnet_correct = np.sum((pointnet_pred_val == label_batch) * weight_batch.astype(np.bool))
        pointnet_total_correct += pointnet_correct
        pointnet_total_seen += np.sum(weight_batch.astype(np.bool)) 
        pointnet_loss_sum += pointnet_loss_val

        avg_ftime += time.time()-start_time
        ## Augment batched point clouds by rotation
        if (batch_idx+1)%10 == 0:
            log_string(' -- %03d / %03d --' % (batch_idx+1, num_batches))
            log_string('average get batch time: %f, average forwarding time: %f' % (avg_btime/ 10, avg_ftime/10))
            log_string('pointnet loss: %f' % (pointnet_loss_sum / 10))
            log_string('pointnet accuracy: %f' % (pointnet_total_correct / float(pointnet_total_seen)))
            pointnet_total_correct, pointnet_total_seen, pointnet_loss_sum = 0, 0, 0
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

    if POINTNET_RESTORE_MODEL !=None:
        pointnet_restore_vars = train_utils.restore_vars(POINTNET_RESTORE_MODEL, 'pointnet2')
        pointnet_loader = tf.train.Saver(var_list=pointnet_restore_vars, max_to_keep=10)
        pointnet_loader.restore(sess, POINTNET_RESTORE_MODEL)
        print ("[INFO] PointNet Model ", POINTNET_RESTORE_MODEL, ' has been restored.')

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
