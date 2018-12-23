import os
import sys
#import collections
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import xception
import train_utils

slim = tf.contrib.slim

def extra_layer_scopes():
    return [
        'logits', 'image_pooling', 'aspp', 
        'concat_projection', 'decoder'
    ]

def joint_encoder(
    images,
    batch_size,
    output_stride=16,
    bn_decay=None,
    is_training=False,
    fine_tune_batch_norm=False):

    # xception root block
    with tf.variable_scope('deeplab') as scope:
        print('*** Images shape:', images.shape) # (480, 640)
        images = (2.0 / 255.0) * tf.to_float(images) - 1.0  # Map image values from [0, 255] to [-1, 1]
        collections = 'deeplab/xception_65/end_points'
        xcept_root = xception.root_block(
                  images, 
                  is_training and fine_tune_batch_norm,
                  collections)
        print('xcepetion/root_block: ', xcept_root.get_shape()) # (240, 320)
    # xception entry_flow/block1
    with tf.variable_scope('deeplab') as scope:
        current_stride = 2 
        atrous_rate = 1
        xcept_enf_b1, atrous_rate, current_stride = \
            xception.get_block(
                xcept_root,
                xception.xception_block(
                    'entry_flow/block1',
                    depth_list=[128, 128, 128],
                    skip_connection_type='conv',
                    activation_fn_in_separable_conv=False,
                    regularize_depthwise=False,
                    num_units=1,
                    stride=2),
                atrous_rate,
                current_stride,
                output_stride,
                is_training and fine_tune_batch_norm,
                collections)
        print('xception/entry_flow/block1: ', xcept_enf_b1.get_shape(), atrous_rate, current_stride) # (120, 160)

    # xception entry_flow/block2 
    with tf.variable_scope('deeplab') as scope:
        xcept_enf_b2, atrous_rate, current_stride = \
            xception.get_block(
                xcept_enf_b1,
                xception.xception_block(
                    'entry_flow/block2',
                    depth_list=[256, 256, 256],
                    skip_connection_type='conv',
                    activation_fn_in_separable_conv=False,
                    regularize_depthwise=False,
                    num_units=1,
                    stride=2),
                atrous_rate,
                current_stride,
                output_stride,
                is_training and fine_tune_batch_norm,
                collections)
        print('xception/entry_flow/block2: ', xcept_enf_b2.get_shape(), atrous_rate, current_stride) # (60, 80)

    # xception entry_flow/block3
    with tf.variable_scope('deeplab') as scope:
        xcept_enf_b3, atrous_rate, current_stride = \
            xception.get_block(
                xcept_enf_b2,
                xception.xception_block(
                    'entry_flow/block3',
                    depth_list=[728, 728, 728],
                    skip_connection_type='conv',
                    activation_fn_in_separable_conv=False,
                    regularize_depthwise=False,
                    num_units=1,
                    stride=2),
                atrous_rate,
                current_stride,
                output_stride,
                is_training and fine_tune_batch_norm,
                collections)
        print('xception/entry_flow/block3: ', xcept_enf_b3.get_shape(), atrous_rate, current_stride) # (30, 40)
    # xception middle_flow
    with tf.variable_scope('deeplab') as scope:
        xcept_mdf_b1, atrous_rate, current_stride = \
            xception.get_block(
                xcept_enf_b3,
                xception.xception_block('middle_flow/block1',
                  depth_list=[728, 728, 728],
                  skip_connection_type='sum',
                  activation_fn_in_separable_conv=False,
                  regularize_depthwise=False,
                  num_units=16,
                  stride=1),
                atrous_rate,
                current_stride,
                output_stride,
                is_training and fine_tune_batch_norm,
                collections)
        print('xception/middle_flow/block3: ', xcept_mdf_b1.get_shape(), atrous_rate, current_stride) # (30, 40)
    # xception exit_flow
    with tf.variable_scope('deeplab') as scope:
        xcept_exf_b1, atrous_rate, current_stride = \
            xception.get_block(
                xcept_mdf_b1,
                xception.xception_block(
                    'exit_flow/block1',
                    depth_list=[728, 1024, 1024],
                    skip_connection_type='conv',
                    activation_fn_in_separable_conv=False,
                    regularize_depthwise=False,
                    num_units=1,
                    stride=2),
                atrous_rate,
                current_stride,
                output_stride,
                is_training and fine_tune_batch_norm,
                collections)
        print('xception/exit_flow/block1: ', xcept_exf_b1.get_shape(), atrous_rate, current_stride) # (30, 40)
        xcept_exf_b2, atrous_rate, current_stride = \
            xception.get_block(
                xcept_exf_b1,
                xception.xception_block('exit_flow/block2',
                  depth_list=[1536, 1536, 2048],
                  skip_connection_type='none',
                  activation_fn_in_separable_conv=True,
                  regularize_depthwise=False,
                  num_units=1,
                  stride=1),
                atrous_rate,
                current_stride,
                output_stride,
                is_training and fine_tune_batch_norm,
                collections)
        print('xception/exit_flow/block2: ', xcept_exf_b2.get_shape(), atrous_rate, current_stride) # (30, 40)
        deeplab_end_points = slim.utils.convert_collection_to_dict(
            collections, clear_collection=True)

    return xcept_exf_b2, deeplab_end_points

def deeplab_aspp_module(
    features,
    image_size,
    output_stride,
    is_training=False,
    fine_tune_batch_norm=False):

    with tf.variable_scope('deeplab') as scope:
        batch_norm_params = {
            'is_training': is_training and fine_tune_batch_norm,
            'decay': 0.9997,
            'epsilon': 1e-5,
            'scale': True,
        }
        with slim.arg_scope(
            [slim.conv2d, slim.separable_conv2d],
            weights_regularizer=slim.l2_regularizer(0.0004),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            padding='SAME',
            stride=1,
            reuse=None):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                depth = 256
                branch_logits = []

                # image level feature 30 x 40 x nd -> 1 x 1 x nd -> 1 x 1 x 256 -> 30 x 40 x256
                pool_height = int((float(image_size[0]) - 1.0) * 1.0 / output_stride + 1.0)
                pool_width = int((float(image_size[1]) - 1.0) * 1.0 / output_stride + 1.0)
                image_feature = slim.avg_pool2d(
                    features, [pool_height, pool_width], [pool_height, pool_width],
                    padding='VALID')
                image_feature = slim.conv2d(
                    image_feature, depth, 1, scope='image_pooling')
                image_feature = tf.image.resize_bilinear(
                    image_feature, [pool_height, pool_width], align_corners=True)
                image_feature.set_shape([None, pool_height, pool_width, depth])
                branch_logits.append(image_feature)

                # xcetipn feature, employ a 1x1 convolution. 30 x 40 x nd -> 30 x 40 x256
                branch_logits.append(slim.conv2d(features, depth, 1,
                                                 scope='aspp' + str(0)))

                # atrous convolution on xception feature only
                # employ 3x3 convolutions with different atrous rates.
                for i, rate in enumerate([6, 12, 18], 1):
                    # aspp with separable conv
                    aspp_features = _split_separable_conv2d(
                        features,
                        filters=depth,
                        rate=rate,
                        weight_decay=0.0004,
                        scope='aspp' + str(i))
                    branch_logits.append(aspp_features)

                # Merge branch logits.
                concat_logits = tf.concat(branch_logits, 3)
                concat_logits = slim.conv2d(
                    concat_logits, depth, 1, scope='concat_projection')
                concat_logits = slim.dropout(
                    concat_logits,
                    keep_prob=0.9,
                    is_training=is_training,
                    scope='concat_projection_dropout')

                return concat_logits 


def _split_separable_conv2d(
    inputs,
    filters,
    rate=1,
    weight_decay=0.00004,
    depthwise_weights_initializer_stddev=0.33,
    pointwise_weights_initializer_stddev=0.06,
    scope=None):
  outputs = slim.separable_conv2d(
      inputs,
      None,
      3,
      depth_multiplier=1,
      rate=rate,
      weights_initializer=tf.truncated_normal_initializer(
          stddev=depthwise_weights_initializer_stddev),
      weights_regularizer=None,
      scope=scope + '_depthwise')
  return slim.conv2d(
      outputs,
      filters,
      1,
      weights_initializer=tf.truncated_normal_initializer(
          stddev=pointwise_weights_initializer_stddev),
      weights_regularizer=slim.l2_regularizer(weight_decay),
      scope=scope + '_pointwise')


def deeplab_decoder(
    features,
    end_points,
    decoder_output_stride,
    image_size,
    bn_decay=None,
    is_training=False,
    fine_tune_batch_norm=False):

    with tf.variable_scope('deeplab') as scope:
        # Scales the input dimension.
        decoder_height = int((float(image_size[0]) - 1.0) * 1.0 / decoder_output_stride + 1.0)
        decoder_width = int((float(image_size[1]) - 1.0) * 1.0 / decoder_output_stride + 1.0)
  
        # decoder batch norm parameters
        batch_norm_params = {
            'is_training': is_training and fine_tune_batch_norm,
            'decay': 0.9997,
            'epsilon': 1e-5,
            'scale': True,
        }
        outer_var_scope_name = tf.get_variable_scope().name
        with slim.arg_scope(
              [slim.conv2d, slim.separable_conv2d],
              weights_regularizer=slim.l2_regularizer(0.0004),
              activation_fn=tf.nn.relu,
              normalizer_fn=slim.batch_norm,
              padding='SAME',
              stride=1,
              reuse=None):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                with tf.variable_scope('decoder', 'decoder', [features]):
                    decoder_features_list = [features]
                    decoder_features_list.append(
                        slim.conv2d(
                            end_points['deeplab/xception_65/entry_flow/block2/unit_1/xception_module/separable_conv2_pointwise'],
                            48,
                            1,
                            scope='feature_projection0'))
                    # Resize to decoder_height/decoder_width.
                    for j, feature in enumerate(decoder_features_list):
                        decoder_features_list[j] = tf.image.resize_bilinear(
                            feature, [decoder_height, decoder_width], align_corners=True)
                        decoder_features_list[j].set_shape(
                            [None, decoder_height, decoder_width, None])
                    decoder_features = tf.concat(decoder_features_list, 3)
                    decoder_features = _split_separable_conv2d(
                        decoder_features,
                        filters=256,
                        rate=1,
                        weight_decay=0.0004,
                        scope='decoder_conv0')
                    decoder_features = _split_separable_conv2d(
                        decoder_features,
                        filters=256,
                        rate=1,
                        weight_decay=0.0004,
                        scope='decoder_conv1')
                return decoder_features

def deeplab_output_layer(
    features,
    num_classes,
    scope_suffix='semantic'):
    with tf.variable_scope('deeplab') as scope:
        with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(0.0001),
            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
            reuse=None):
          with tf.variable_scope('logits', 'logits', [features]):
              scope = scope_suffix + '_0'
              logits =  slim.conv2d(
                            features,
                            num_classes,
                            kernel_size=1,
                            rate=1,
                            activation_fn=None,
                            normalizer_fn=None,
                            scope=scope)
              return logits


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

