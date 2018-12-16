import sys
import os
import time
import tensorflow as tf
import numpy as np
from collections import OrderedDict
from tensorflow.python import pywrap_tensorflow

import build_deeplab 


NUM_CLASSES = 21
INIT_CHECK_POINT_PATH = '../../deeplabv3_xception_ade20k_train/model.ckpt'#your ckpt path
PREFIX = 'deeplab'

'''
    source from : https://www.jianshu.com/p/4e53d3c604f6
'''
def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(low=low, high=high, size=(fan_in, fan_out))


def build_graph():
    is_training = True 
    image = tf.placeholder(dtype=tf.float32, shape=(1, 480, 640, 3))
    label = tf.placeholder(dtype=tf.int32, shape=(1, 480, 640, 1))
    outputs_to_num_classes = {'semantic':NUM_CLASSES}
    model_options = build_deeplab.model_option(outputs_to_num_classes)
    with tf.variable_scope('deeplab') as scope:
        features, end_points = build_deeplab.build_encoder(
                                   image_batch=image,
                                   model_options=model_options,
                                   is_training=is_training,
                                   fine_tune_bn=False)
    with tf.variable_scope('deeplab') as scope:
        features = build_deeplab.build_decoder(
                       features,
                       end_points,
                       model_options=model_options,
                       is_training=is_training,
                       fine_tune_bn=False)
    with tf.variable_scope('deeplab') as scope:
        logit = build_deeplab.build_output_layer(
                     features, 
                     NUM_CLASSES,
                     model_options)

def make_init():
    checkpoint_path = INIT_CHECK_POINT_PATH
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    var_to_shape_map = OrderedDict(sorted(var_to_shape_map.items()))
    deeplabv3={}
    for key, shape in var_to_shape_map.items():
        str_name = key
        if str_name.find('Momentum') > -1:
            continue
        deeplabv3[PREFIX+'/'+str_name] = reader.get_tensor(key)
        print ("tensor_name", PREFIX+'/'+str_name, ', shape:', shape, ', tensor array:', reader.get_tensor(key).shape)
    return deeplabv3
    
def run():
    deeplabv3_weights = make_init()
    build_graph()

    #Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    global_vars = tf.global_variables()
    for i in range(len(global_vars)):
        if 'logits/semantic/weights' in global_vars[i].name:
            print('Logit Weights using Xavier initializer from numpy')
            logits_shape = global_vars[i].get_shape().as_list()
            w = xavier_init(logits_shape[2], logits_shape[3]).astype(np.float32) 
            w = np.expand_dims(np.expand_dims(w, axis=0), axis=0)
            print(global_vars[i].name, w.shape)
            global_vars[i].load(w, session=sess)
        elif 'logits/semantic/biases' in global_vars[i].name:
            print('Logit Biases using zero initializer')
            logits_shape = global_vars[i].get_shape()
            w = np.zeros(logits_shape, dtype=np.float32) 
            print(global_vars[i].name, w.shape)
            global_vars[i].load(w, session=sess)
        else:
            for k, w in deeplabv3_weights.items():
                if k in global_vars[i].name:
                    print(global_vars[i], k, w.shape)
                    global_vars[i].load(w, session=sess)
    saver.save(sess, './init_weights/deeplabv3_xception_init')

if __name__ == "__main__":
    run()

