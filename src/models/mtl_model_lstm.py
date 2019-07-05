import tensorflow as tf
from src.models.base_model import *
from src.inputs.readLiarFile import *

FLAGS = tf.app.flags.FLAGS

TASK_NUM = 2

class MTLModel(BaseModel):

  def __init__(self, task_labels, labels, sentences, adv, is_train):
    self.is_train = is_train
    self.adv = adv
    self.word_dim = 200
    w_trainable = True if self.word_dim == 200 else False
    self.shared_conv = ConvLayer('conv_shared', FILTER_SIZES)
    self.shared_linear = LinearLayer('linear_shared', TASK_NUM, True)
    self.tensors = []

    self.n_hidden = 100
    self.n_classes = 6
    # Define weights
    self.weights = {
      # Hidden layer weights => 2*n_hidden because of foward + backward cells
      'out': tf.Variable(tf.random_normal([2 * self.n_hidden, self.n_classes]))
    }
    self.biases = {
      'out': tf.Variable(tf.random_normal([self.n_classes]))
    }

    self.build_task_graph(task_labels, labels, sentences)

  def adversarial_loss(self, feature, task_label):
    ''' make the task classifier cannot reliably predict the task based on the shared feature
    '''
    feature = flip_gradient(feature)
    if self.is_train:
      feature = tf.nn.dropout(feature, FLAGS.keep_prob)
    # Map the features to TASK_NUM classes
    logits, loss_l2 = self.shared_linear(feature)
    loss_adv = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=task_label, logits=logits))
    return loss_adv, loss_l2
  
  def diff_loss(self, shared_feat, task_feat):
    # print('diff loss ...')
    '''Orthogonality Constraints from https://github.com/tensorflow/models,
    in directory research/domain_adaptation
    '''
    task_feat -= tf.reduce_mean(task_feat, 0)
    shared_feat -= tf.reduce_mean(shared_feat, 0)

    task_feat = tf.nn.l2_normalize(task_feat, 1)
    shared_feat = tf.nn.l2_normalize(shared_feat, 1)

    correlation_matrix = tf.matmul(
        task_feat, shared_feat, transpose_a=True)

    cost = tf.reduce_mean(tf.square(correlation_matrix)) * 0.01
    cost = tf.where(cost > 0, cost, 0, name='value')

    assert_op = tf.Assert(tf.is_finite(cost), [cost])
    with tf.control_dependencies([assert_op]):
      loss_diff = tf.identity(cost)

    return loss_diff

  def build_task_graph(self, task_labels, labels, sentences):
    sentences = tf.convert_to_tensor(sentences)
    if self.is_train:
      sentences = tf.nn.dropout(sentences, FLAGS.keep_prob)

    sentences = tf.transpose(sentences, [1, 0, 2])
    sentences = tf.reshape(sentences, [-1, 200])
    sentences = tf.split(sentences, 30)

    with tf.variable_scope('forward'):
      lstm_qx = tf.contrib.rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)
    with tf.variable_scope('backward'):
      lstm_hx = tf.contrib.rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)

    with tf.variable_scope('shared_out'):
      shared_outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_qx, lstm_hx, sentences, dtype=tf.float32)
    shared_outputs = shared_outputs[-1]
    print('output:::')
    print(shared_outputs)
    print(shared_outputs[-1])
    pred = tf.matmul(shared_outputs, self.weights['out']) + self.biases['out']
    print('pred:')
    print(pred)
    y = []
    for i in labels:
      y.append(i[0])
    y = tf.convert_to_tensor(y)
    print('shuchu y:')
    print(y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accurancy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    self.tensors.append((accurancy, cost))

  def build_train_op(self):
    if self.is_train:
      self.train_ops = []
      for _, loss in self.tensors:
        train_op = optimize(loss)
        self.train_ops.append(train_op)

def build_train_valid_model(task_labels, labels, sentences, adv, test):
  if test == False:
    with tf.name_scope("Train"):
        m_train = MTLModel(task_labels, labels, sentences, adv, is_train=True)
        if not test:
          m_train.build_train_op()
  # if test == True:
  #   with tf.name_scope('Valid'):
  #       print('valid...')
  #       m_train = MTLModel(task_labels, labels, sentences, adv, is_train=True)
  # m_train.build_train_op()
    # return m_train, m_valid
  return m_train