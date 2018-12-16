import tensorflow as tf
from collections import OrderedDict
from tensorflow.python.framework import ops
from tensorflow.python.framework import constant_op
from tensorflow.python import pywrap_tensorflow

slim = tf.contrib.slim

def model_learning_rate(
    global_step,
    learning_policy, base_learning_rate, learning_rate_decay_step,
    learning_rate_decay_factor, training_number_of_steps, learning_power,
    slow_start_step, slow_start_learning_rate):
  if learning_policy == 'step':
    learning_rate = tf.train.exponential_decay(
        base_learning_rate,
        global_step,
        learning_rate_decay_step,
        learning_rate_decay_factor,
        staircase=True)
  elif learning_policy == 'poly':
    learning_rate = tf.train.polynomial_decay(
        base_learning_rate,
        global_step,
        training_number_of_steps,
        end_learning_rate=0,
        power=learning_power)
  else:
    raise ValueError('Unknown learning policy.')

  # Employ small learning rate at the first few steps for warm start.
  learing_rate = tf.where(global_step < slow_start_step, slow_start_learning_rate,
                  learning_rate)
  learing_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
  return learning_rate

def get_bn_decay(
    global_step,
    bn_init_decay,
    bn_decay_step,
    bn_decay_rate,
    bn_decay_clip
    ):
    bn_momentum = tf.train.exponential_decay(
                      bn_init_decay,
                      global_step,
                      bn_decay_step,
                      bn_decay_rate,
                      staircase=True)
    bn_decay = tf.minimum(bn_decay_clip, 1 - bn_momentum)
    return bn_decay

def model_gradient_multipliers(last_layers, last_layer_gradient_multiplier, grad_vars):
  gradient_multipliers = {}

  for gv in grad_vars:
    # Double the learning rate for biases.
    if 'biases' in gv[1].name:
      gradient_multipliers[gv[1].name] = 2.

    # Use larger learning rate for last layer variables.
    for layer in last_layers:
      if layer in gv[1].name and 'biases' in gv[1].name:
        gradient_multipliers[gv[1].name] = 2 * last_layer_gradient_multiplier
        break
      elif layer in gv[1].name:
        gradient_multipliers[gv[1].name] = last_layer_gradient_multiplier
        break

  return gradient_multipliers


def multiply_gradients(grads_and_vars, gradient_multipliers):
  if not isinstance(grads_and_vars, list):
    raise ValueError('`grads_and_vars` must be a list.')
  if not gradient_multipliers:
    raise ValueError('`gradient_multipliers` is empty.')
  if not isinstance(gradient_multipliers, dict):
    raise ValueError('`gradient_multipliers` must be a dict.')

  multiplied_grads_and_vars = []
  for grad, var in grads_and_vars:
    if var in gradient_multipliers or var.op.name in gradient_multipliers:
      key = var if var in gradient_multipliers else var.op.name
      if grad is None:
        raise ValueError('Requested multiple of `None` gradient.')

      multiplier = gradient_multipliers[key]
      if not isinstance(multiplier, ops.Tensor):
        multiplier = constant_op.constant(multiplier, dtype=grad.dtype)

      if isinstance(grad, ops.IndexedSlices):
        tmp = grad.values * multiplier
        grad = ops.IndexedSlices(tmp, grad.indices, grad.dense_shape)
      else:
        grad *= multiplier
    multiplied_grads_and_vars.append((grad, var))
  return multiplied_grads_and_vars



def average_gradients(tower_grads):
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def get_smidx(batch_size, input_size, bottle_size):
    #in_height = layer_input.get_shape()[1]
    #in_width = layer_input.get_shape()[2]
    in_height = input_size[0]
    in_width = input_size[1]
    btl_height, btl_width = bottle_size[0], bottle_size[1]
    assert in_height % btl_height == 0 and in_width % btl_width == 0 \
    , "input height, width: {}, {}; bottle neck height, width: {}, {}" \
    .format(in_height, in_width, btl_height, btl_width)
    assert in_height // btl_height ==  in_width // btl_width \
    , "input height, width: {}, {}; bottle neck height, width: {}, {}" \
    .format(in_height, in_width, btl_height, btl_width)
    if in_height > btl_height and in_width > btl_width:
        hidx = tf.range(1, in_height+1, 2)
        widx = tf.range(1, in_width+1, 2)
    else:
        hidx = tf.range(0, in_height, 1)
        widx = tf.range(0, in_width, 1)
    wv, hv = tf.meshgrid(widx, hidx)
    smidx = hv*in_width + wv
    output_shape = (smidx.get_shape()[0], smidx.get_shape()[1])

    smidx = tf.expand_dims(tf.reshape(smidx, [-1]), axis=0)
    smidx = tf.tile(smidx, [batch_size, 1])
    return smidx, output_shape


def model_restore_save_vars(model_scope):
    # Get Model restore and save variables
    print('[INFO] ------ Getting Variables Under:', model_scope, ' ------\n')
    global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model_scope)
    print('[INFO] Length of Global Variables:\n', len(global_vars), '\n--\n')
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=model_scope)
    trainable_vars = [t_vars for t_vars in trainable_vars \
                                 if ('BatchNorm' not in t_vars.name) and \
                                    ('gamma' not in t_vars.name) and \
                                    ('beta' not in t_vars.name)]
    print('[INFO] Length of Trainable Variables:\n', len(trainable_vars), '\n--\n')
    bn_vars = [bn for bn in global_vars \
                          if ('Momentum' not in bn.name) and \
                             ('gamma' in bn.name or \
                              'beta' in bn.name or \
                              'moving_mean' in bn.name or \
                              'moving_variance' in bn.name)]
    print('[INFO] Length of BatchNorm Variables:\n', len(bn_vars), '\n--\n')
    trainable_bn_vars = trainable_vars + bn_vars 
    print('[INFO] Length of Trainable and BatchNorm Variables:\n', len(trainable_bn_vars), '\n--\n')
    restore_vars_except_logit = [v for v in trainable_bn_vars if 'logits' not in v.name]
    print('[INFO] Length of Global Variables Except Logits:\n', len(restore_vars_except_logit), '\n--\n')
    
    return restore_vars_except_logit, trainable_bn_vars


def restore_vars(restore_model, model_scope):
    # read restore_model key
    reader = pywrap_tensorflow.NewCheckpointReader(restore_model)
    restore_vars_to_shape_map = reader.get_variable_to_shape_map()
    restore_vars_to_shape_map = OrderedDict(sorted(restore_vars_to_shape_map.items()))
    # Get Model restore and save variables
    print('[INFO] Getting Variables Under:', model_scope)
    global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model_scope)
    # get restore variables
    restore_vars = list()
    for v in global_vars:
        # get rid of ':0' in var name
        if v.name[:-2] in restore_vars_to_shape_map:
            restore_vars.append(v)
    print('[INFO] Restore {} Variables Under {}'.format(len(restore_vars), restore_model))
    return restore_vars



