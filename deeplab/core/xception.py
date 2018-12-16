import collections
import tensorflow as tf

from tensorflow.contrib.slim.nets import resnet_utils

slim = tf.contrib.slim


_DEFAULT_MULTI_GRID = [1, 1, 1]


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
  """A named tuple describing an Xception block.

  Its parts are:
    scope: The scope of the block.
    unit_fn: The Xception unit function which takes as input a tensor and
      returns another tensor with the output of the Xception unit.
    args: A list of length equal to the number of units in the block. The list
      contains one dictionary for each unit in the block to serve as argument to
      unit_fn.
  """


def fixed_padding(inputs, kernel_size, rate=1):
  kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
  pad_total = kernel_size_effective - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                  [pad_beg, pad_end], [0, 0]])
  return padded_inputs



@slim.add_arg_scope
def separable_conv2d_same(inputs,
                          num_outputs,
                          kernel_size,
                          depth_multiplier,
                          stride,
                          rate=1,
                          use_explicit_padding=True,
                          regularize_depthwise=False,
                          scope=None,
                          **kwargs):
  def _separable_conv2d(padding):
    """Wrapper for separable conv2d."""
    return slim.separable_conv2d(inputs,
                                 num_outputs,
                                 kernel_size,
                                 depth_multiplier=depth_multiplier,
                                 stride=stride,
                                 rate=rate,
                                 padding=padding,
                                 scope=scope,
                                 **kwargs)
  def _split_separable_conv2d(padding):
    """Splits separable conv2d into depthwise and pointwise conv2d."""
    outputs = slim.separable_conv2d(inputs,
                                    None,
                                    kernel_size,
                                    depth_multiplier=depth_multiplier,
                                    stride=stride,
                                    rate=rate,
                                    padding=padding,
                                    scope=scope + '_depthwise',
                                    **kwargs)
    return slim.conv2d(outputs,
                       num_outputs,
                       1,
                       scope=scope + '_pointwise',
                       **kwargs)
  if stride == 1 or not use_explicit_padding:
    if regularize_depthwise:
      outputs = _separable_conv2d(padding='SAME')
    else:
      outputs = _split_separable_conv2d(padding='SAME')
  else:
    inputs = fixed_padding(inputs, kernel_size, rate)
    if regularize_depthwise:
      outputs = _separable_conv2d(padding='VALID')
    else:
      outputs = _split_separable_conv2d(padding='VALID')
  return outputs


@slim.add_arg_scope
def xception_module(inputs,
                    depth_list,
                    skip_connection_type,
                    stride,
                    unit_rate_list=None,
                    rate=1,
                    activation_fn_in_separable_conv=False,
                    regularize_depthwise=False,
                    outputs_collections=None,
                    scope=None):
  if len(depth_list) != 3:
    raise ValueError('Expect three elements in depth_list.')
  if unit_rate_list:
    if len(unit_rate_list) != 3:
      raise ValueError('Expect three elements in unit_rate_list.')

  with tf.variable_scope(scope, 'xception_module', [inputs]) as sc:
    residual = inputs

    def _separable_conv(features, depth, kernel_size, depth_multiplier,
                        regularize_depthwise, rate, stride, scope):
      if activation_fn_in_separable_conv:
        activation_fn = tf.nn.relu
      else:
        activation_fn = None
        features = tf.nn.relu(features)
      return separable_conv2d_same(features,
                                   depth,
                                   kernel_size,
                                   depth_multiplier=depth_multiplier,
                                   stride=stride,
                                   rate=rate,
                                   activation_fn=activation_fn,
                                   regularize_depthwise=regularize_depthwise,
                                   scope=scope)
    for i in range(3):
      residual = _separable_conv(residual,
                                 depth_list[i],
                                 kernel_size=3,
                                 depth_multiplier=1,
                                 regularize_depthwise=regularize_depthwise,
                                 rate=rate*unit_rate_list[i],
                                 stride=stride if i == 2 else 1,
                                 scope='separable_conv' + str(i+1))
    if skip_connection_type == 'conv':
      shortcut = slim.conv2d(inputs,
                             depth_list[-1],
                             [1, 1],
                             stride=stride,
                             activation_fn=None,
                             scope='shortcut')
      outputs = residual + shortcut
    elif skip_connection_type == 'sum':
      outputs = residual + inputs
    elif skip_connection_type == 'none':
      outputs = residual
    else:
      raise ValueError('Unsupported skip connection type.')

    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.name,
                                            outputs)

@slim.add_arg_scope
def root_block(
    inputs, 
    is_training,
    collections):

    arg_scope = xception_arg_scope()
    # Extract features for entry_flow, middle_flow, and exit_flow.
    with slim.arg_scope(arg_scope):
        with tf.variable_scope('xception_65', 'xception') as sc:
            with slim.arg_scope([slim.conv2d,
                                 slim.separable_conv2d,
                                 xception_module,
                                 get_block,
                                 root_block],
                                 outputs_collections=collections):
                # Root block function operated on inputs.
                with slim.arg_scope([slim.batch_norm], is_training=is_training):
                    net = inputs
                    print(net.get_shape())
                    net = resnet_utils.conv2d_same(net, 32, 3, stride=2,
                                                   scope='entry_flow/conv1_1')
                    print(net.get_shape())
                    net = resnet_utils.conv2d_same(net, 64, 3, stride=1,
                                                   scope='entry_flow/conv1_2')
                    print(net.get_shape())
    return net


@slim.add_arg_scope
def get_block(
    net,
    block,
    atrous_rate,
    current_stride,
    output_stride,
    is_training,
    collections):
    arg_scope = xception_arg_scope()
    with slim.arg_scope(arg_scope):
        with tf.variable_scope('xception_65', 'xception') as sc:
            with slim.arg_scope([slim.conv2d,
                                 slim.separable_conv2d,
                                 xception_module,
                                 get_block,
                                 root_block],
                                 outputs_collections=collections):
                with slim.arg_scope([slim.batch_norm], is_training=is_training):
                    with tf.variable_scope(block.scope, 'block', [net]) as sc:
                        for i, unit in enumerate(block.args):
                            with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                                if output_stride is not None and current_stride == output_stride:
                                  net = block.unit_fn(net, rate=atrous_rate, **dict(unit, stride=1))
                                  atrous_rate *= unit.get('stride', 1)
                                else:
                                  net = block.unit_fn(net, rate=1, **unit)
                                  current_stride *= unit.get('stride', 1)
    return net, atrous_rate, current_stride



def xception_block(scope,
                   depth_list,
                   skip_connection_type,
                   activation_fn_in_separable_conv,
                   regularize_depthwise,
                   num_units,
                   stride,
                   unit_rate_list=None):
  if unit_rate_list is None:
    unit_rate_list = _DEFAULT_MULTI_GRID
  return Block(scope, xception_module, [{
      'depth_list': depth_list,
      'skip_connection_type': skip_connection_type,
      'activation_fn_in_separable_conv': activation_fn_in_separable_conv,
      'regularize_depthwise': regularize_depthwise,
      'stride': stride,
      'unit_rate_list': unit_rate_list,
  }] * num_units)



def xception_arg_scope(weight_decay=0.00004,
                       batch_norm_decay=0.9997,
                       batch_norm_epsilon=0.001,
                       batch_norm_scale=True,
                       weights_initializer_stddev=0.09,
                       activation_fn=tf.nn.relu,
                       regularize_depthwise=False,
                       use_batch_norm=True):
  """Defines the default Xception arg scope.

  Args:
    weight_decay: The weight decay to use for regularizing the model.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    batch_norm_epsilon: Small constant to prevent division by zero when
      normalizing activations by their variance in batch normalization.
    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
      activations in the batch normalization layer.
    weights_initializer_stddev: The standard deviation of the trunctated normal
      weight initializer.
    activation_fn: The activation function in Xception.
    regularize_depthwise: Whether or not apply L2-norm regularization on the
      depthwise convolution weights.
    use_batch_norm: Whether or not to use batch normalization.

  Returns:
    An `arg_scope` to use for the Xception models.
  """
  batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
  }
  if regularize_depthwise:
    depthwise_regularizer = slim.l2_regularizer(weight_decay)
  else:
    depthwise_regularizer = None
  with slim.arg_scope(
      [slim.conv2d, slim.separable_conv2d],
      weights_initializer=tf.truncated_normal_initializer(
          stddev=weights_initializer_stddev),
      activation_fn=activation_fn,
      normalizer_fn=slim.batch_norm if use_batch_norm else None):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope(
          [slim.conv2d],
          weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope(
            [slim.separable_conv2d],
            weights_regularizer=depthwise_regularizer) as arg_sc:
          return arg_sc
