import os
import sys
import collections
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
sys.path.append(BASE_DIR)
import tf_util
import train_utils
from pointnet_util import pointnet_sa_module, pointnet_fp_module

sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops/3d_interpolation'))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point
from tf_interpolate import three_nn, three_interpolate

slim = tf.contrib.slim

def interpolate_points(dense_xyz, sparse_xyz, sparse_point):
    dist, idx = three_nn(dense_xyz, sparse_xyz)
    dist = tf.maximum(dist, 1e-10)
    norm = tf.reduce_sum((1.0/dist),axis=2,keep_dims=True)
    norm = tf.tile(norm,[1,1,3])
    weight = (1.0/dist) / norm
    interpolated_points = three_interpolate(sparse_point, idx, weight)
    return interpolated_points


def pointnet2(point_cloud, scene_point, is_training, num_class, bn_decay=None):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    with tf.variable_scope('layer_0') as sc:
        l0_v_xyz = point_cloud[...,0:3]
        l0_v_points = tf.concat([point_cloud[...,3:6], point_cloud[..., 9:]], axis=-1)
        l0_s_xyz = scene_point[...,0:3]
        l0_s_points = tf.concat([scene_point[...,3:6], scene_point[..., 9:]], axis=-1)
        end_points['l0_v_xyz'] = l0_v_xyz
        end_points['l0_s_xyz'] = l0_s_xyz

    with tf.variable_scope('sa_layer_1') as sc:
        l1_v_xyz, l1_v_points, l1_v_indices = pointnet_sa_module(l0_v_xyz, l0_v_points, npoint=1024, radius=0.1, nsample=32, mlp=[256, 256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='volume_layer1')
        l1_s_xyz, l1_s_points, l1_s_indices = pointnet_sa_module(l0_s_xyz, l0_s_points, npoint=4096, radius=0.4, nsample=32, mlp=[256, 256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='scene_layer1')


    with tf.variable_scope('sa_layer_2') as sc:
        l2_v_xyz, l2_v_points, l2_v_indices = pointnet_sa_module(l1_v_xyz, l1_v_points, npoint=256, radius=0.2, nsample=32, mlp=[256, 256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='volume_layer2')
        l2_s_xyz, l2_s_points, l2_s_indices = pointnet_sa_module(l1_s_xyz, l1_s_points, npoint=1024, radius=0.8, nsample=32, mlp=[256, 256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='scene_layer2')


    with tf.variable_scope('sa_layer_3') as sc:
        l3_v_xyz, l3_v_points, l3_v_indices = pointnet_sa_module(l2_v_xyz, l2_v_points, npoint=64, radius=0.4, nsample=32, mlp=[512, 512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='volume_layer3')
        l3_s_xyz, l3_s_points, l3_s_indices = pointnet_sa_module(l2_s_xyz, l2_s_points, npoint=256, radius=1.2, nsample=32, mlp=[512, 512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='scene_layer3')

    with tf.variable_scope('sa_layer_4') as sc:
        l4_v_xyz, l4_v_points, l4_v_indices = pointnet_sa_module(l3_v_xyz, l3_v_points, npoint=16, radius=0.8, nsample=32, mlp=[512, 726,726], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='volume_layer4')
        l4_s_xyz, l4_s_points, l4_s_indices = pointnet_sa_module(l3_s_xyz, l3_s_points, npoint=128, radius=1.6, nsample=32, mlp=[512, 726, 726], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='scene_layer4')

    # Feature Propagation layers
    with tf.variable_scope('fp_layer_1') as sc:
        mlp = [256, 256]
        l4_v_interp_points = interpolate_points(l3_v_xyz, l4_v_xyz, l4_v_points)
        l4_s_interp_points = interpolate_points(l3_v_xyz, l4_s_xyz, l4_s_points)

        fp_feature = tf.concat([l4_v_interp_points, l4_s_interp_points, l3_v_points], axis=-1) # B,ndataset1,nchannel1+nchannel2
        fp_feature = tf.expand_dims(fp_feature, 2)
        for i, num_out_channel in enumerate(mlp):
            fp_feature = tf_util.conv2d(fp_feature, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=True, is_training=is_training,
                                        scope='conv_%d'%(i), bn_decay=bn_decay)
        l3_fp_points = tf.squeeze(fp_feature, [2]) # B,ndataset1,mlp[-1]

    # Feature Propagation layers
    with tf.variable_scope('fp_layer_2') as sc:
        mlp = [256, 256]
        l3_v_interp_points = interpolate_points(l2_v_xyz, l3_v_xyz, l3_fp_points)
        l3_s_interp_points = interpolate_points(l2_v_xyz, l3_s_xyz, l3_s_points)

        fp_feature = tf.concat([l3_v_interp_points, l3_s_interp_points, l2_v_points], axis=-1) # B,ndataset1,nchannel1+nchannel2
        fp_feature = tf.expand_dims(fp_feature, 2)
        for i, num_out_channel in enumerate(mlp):
            fp_feature = tf_util.conv2d(fp_feature, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=True, is_training=is_training,
                                        scope='conv_%d'%(i), bn_decay=bn_decay)
        l2_fp_points = tf.squeeze(fp_feature, [2]) # B,ndataset1,mlp[-1]

    # Feature Propagation layers
    with tf.variable_scope('fp_layer_3') as sc:
        mlp = [256, 256]
        l2_v_interp_points = interpolate_points(l1_v_xyz, l2_v_xyz, l2_fp_points)
        l2_s_interp_points = interpolate_points(l1_v_xyz, l2_s_xyz, l2_s_points)

        fp_feature = tf.concat([l2_v_interp_points, l2_s_interp_points, l1_v_points], axis=-1) # B,ndataset1,nchannel1+nchannel2
        fp_feature = tf.expand_dims(fp_feature, 2)
        for i, num_out_channel in enumerate(mlp):
            fp_feature = tf_util.conv2d(fp_feature, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=True, is_training=is_training,
                                        scope='conv_%d'%(i), bn_decay=bn_decay)
        l1_fp_points = tf.squeeze(fp_feature, [2]) # B,ndataset1,mlp[-1]
    # Feature Propagation layers
    with tf.variable_scope('fp_layer_4') as sc:
        mlp = [256, 256]
        l1_v_interp_points = interpolate_points(l0_v_xyz, l1_v_xyz, l1_fp_points)
        l1_s_interp_points = interpolate_points(l0_v_xyz, l1_s_xyz, l1_s_points)

        fp_feature = tf.concat([l1_v_interp_points, l1_s_interp_points, l0_v_points], axis=-1) # B,ndataset1,nchannel1+nchannel2
        fp_feature = tf.expand_dims(fp_feature, 2)
        for i, num_out_channel in enumerate(mlp):
            fp_feature = tf_util.conv2d(fp_feature, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=True, is_training=is_training,
                                        scope='conv_%d'%(i), bn_decay=bn_decay)
        l0_fp_points = tf.squeeze(fp_feature, [2]) # B,ndataset1,mlp[-1]

    # FC layers
    net = tf_util.conv1d(l0_fp_points, 256, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    end_points['feats'] = net 
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net, num_class, 1, padding='VALID', activation_fn=None, scope='fc2')

    return net, end_points

def get_loss(logits, labels, num_classes, ignore_label, weights, scope='loss'):

  print('*** Shape to calcuate loss:', logits.get_shape(), labels.get_shape(), weights.get_shape())
  not_ignore_mask = tf.to_float(tf.not_equal(labels,
                                             ignore_label))
  weights = weights * not_ignore_mask

  weights_1D = tf.reshape(weights, shape=[-1])
  labels_1D = tf.reshape(labels, shape=[-1])
  logits_1D = tf.reshape(logits, shape=[-1, num_classes])
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels_1D, logits=logits_1D, weights=weights_1D)
  return loss
