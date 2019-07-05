import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

flags = tf.app.flags
flags.DEFINE_string("logdir", "saved_models/", "where to save the model")
flags.DEFINE_integer("num_filters", 100, "cnn number of output unit")
flags.DEFINE_float("lrn_rate", 0.01, "learning rate")
flags.DEFINE_float("l2_coef", 0.01, "l2 loss coefficient")
flags.DEFINE_float("keep_prob", 0.5, "dropout keep probability")

FLAGS = tf.app.flags.FLAGS

FILTER_SIZES = [3, 4, 5]

class BaseModel(object):

  def set_saver(self, save_dir):
    '''
    Args:
      save_dir: relative path to FLAGS.logdir
    '''
    # shared between train and valid model instance
    self.saver = tf.train.Saver(var_list=None)
    self.save_dir = os.path.join(FLAGS.logdir, save_dir)
    self.save_path = os.path.join(self.save_dir, "model.ckpt")

  def restore(self, session):
    ckpt = tf.train.get_checkpoint_state(self.save_dir)
    self.saver.restore(session, ckpt.model_checkpoint_path)

class LinearLayer(tf.layers.Layer):
  '''inherit tf.layers.Layer to cache trainable variables
  '''
  def __init__(self, layer_name, out_size, is_regularize, **kwargs):
    self.layer_name = layer_name
    self.out_size = out_size
    self.is_regularize = is_regularize
    super(LinearLayer, self).__init__(**kwargs)
  
  def build(self, input_shape):
    in_size = input_shape[1]

    with tf.variable_scope(self.layer_name):
      w_init = tf.truncated_normal_initializer(stddev=0.1)
      b_init = tf.constant_initializer(0.1)

      self.w = self.add_variable('W', [in_size, self.out_size], initializer=w_init)
      self.b = self.add_variable('b', [self.out_size], initializer=b_init)

      super(LinearLayer, self).build(input_shape)

  def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
  def call(self, x):
    x_list_size = x.shape.as_list()
    y_size = 1
    for i in x_list_size:
      y_size *= i
    loss_l2 = tf.constant(0, dtype=tf.float32)
    # y_size = [-1, 15 * 100 * 32]
    x = tf.reshape(x, [-1, y_size])
    w = tf.Variable(tf.truncated_normal([y_size, self.out_size], stddev=0.1))
    b = tf.Variable(tf.truncated_normal([self.out_size], stddev=0.1))
    o = tf.nn.xw_plus_b(x, w, b)
    if self.is_regularize:
      loss_l2 += tf.nn.l2_loss(w) + tf.nn.l2_loss(b)
    return o, loss_l2


class ConvLayer(tf.layers.Layer):
  '''inherit tf.layers.Layer to cache trainable variables
  '''
  def __init__(self, layer_name, filter_sizes, **kwargs):
    self.layer_name = layer_name
    self.filter_sizes = filter_sizes
    self.conv = {} # trainable variables for conv
    super(ConvLayer, self).__init__(**kwargs)
    # print("--init--")
  
  def build(self, input_shape):
    input_dim = 200

    with tf.variable_scope(self.layer_name):
      w_init = tf.truncated_normal_initializer(stddev=0.1)
      b_init = tf.constant_initializer(0.1)

      for fsize in self.filter_sizes:
        w_shape = [fsize, input_dim, 1, FLAGS.num_filters]
        b_shape = [FLAGS.num_filters]
        w_name = 'conv-W%d' % fsize
        b_name = 'conv-b%d' % fsize
        self.conv[w_name] = self.add_variable(
                                           w_name, w_shape, initializer=w_init)
        self.conv[b_name] = self.add_variable(
                                           b_name, b_shape, initializer=b_init)
    
      super(ConvLayer, self).build(input_shape)

  def call(self, x):
    # print('--new call--')
    x = tf.reshape(x, [-1, 30, 200, 1])
    filter = tf.Variable(tf.random_normal([5, 5, 1, 64]))
    b_conv1 = tf.Variable(tf.random_normal([64]))
    conv = tf.nn.conv2d(x, filter, strides=[1, 1, 1, 1], padding='SAME')
    conv = tf.nn.relu(conv + b_conv1)
    return conv

def max_pool(conv_outs, max_len):
  pool = tf.nn.max_pool(conv_outs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  return pool

def optimize(loss):
  optimizer = tf.train.AdamOptimizer(FLAGS.lrn_rate)

  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops): # for batch_norm
    train_op = optimizer.minimize(loss)
  return train_op

class FlipGradientBuilder(object):
  '''Gradient Reversal Layer from https://github.com/pumpikano/tf-dann'''
  def __init__(self):
    self.num_calls = 0

  def __call__(self, x, l=1.0):
    grad_name = "FlipGradient%d" % self.num_calls
    @ops.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
      return [tf.negative(grad) * l]
    
    g = tf.get_default_graph()
    with g.gradient_override_map({"Identity": grad_name}):
      y = tf.identity(x)
        
    self.num_calls += 1
    return y
    
flip_gradient = FlipGradientBuilder()