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
    for task_label, label, sentence in zip(task_labels, labels, sentences):
      self.build_task_graph(task_label, label, sentence)

  def adversarial_loss(self, feature, task_label):
    ''' make the task classifier cannot reliably predict the task based on the shared feature
    '''
    feature = flip_gradient(feature)
    if self.is_train:
      feature = tf.nn.dropout(feature, FLAGS.keep_prob)
    # Map the features to TASK_NUM classes
    logits, loss_l2 = self.shared_linear(feature)
    # task_label: true and false
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

  def build_task_graph(self, task_label, label, sentence):
    sentence = tf.convert_to_tensor(sentence)
    # if self.is_train:
    #   sentence = tf.nn.dropout(sentence, FLAGS.keep_prob)
    conv_layer = ConvLayer('conv_task', FILTER_SIZES)
    conv_out = conv_layer(sentence)
    conv_out = max_pool(conv_out, 300)

    shared_out = self.shared_conv(sentence)
    shared_out = max_pool(shared_out, 300)

    if self.adv:
      feature = tf.concat([conv_out, shared_out], axis=1)
    else:
      feature = conv_out

    if self.is_train:
      feature = tf.nn.dropout(feature, FLAGS.keep_prob)

    # Map the features to 6 classes
    linear = LinearLayer('linear', 6, True)
    logits, loss_l2 = linear(feature)
    # class labels label[0]: [0,1,0,0,0,0]
    xentropy = tf.nn.softmax_cross_entropy_with_logits(
                          labels=label[0],
                          logits=logits)
    # six classes
    loss_ce = tf.reduce_mean(xentropy)
    # shared
    loss_adv, loss_adv_l2 = self.adversarial_loss(shared_out, task_label)
    loss_diff = self.diff_loss(shared_out, conv_out)
    if self.adv:

      loss = loss_ce + 0.05*loss_adv + FLAGS.l2_coef*(loss_l2+loss_adv_l2) + loss_diff
    else:
      loss = loss_ce + FLAGS.l2_coef*loss_l2

    pred = tf.argmax(logits, axis=1)

    acc = tf.cast(tf.equal(pred, label[0]), tf.float32)
    acc = tf.reduce_mean(acc)
    self.tensors.append((acc, loss))
    
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

  return m_train