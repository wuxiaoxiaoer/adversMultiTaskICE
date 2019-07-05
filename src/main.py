import os
import time
import sys
import tensorflow as tf
import numpy as np

from src.models import mtl_model
from gensim.models import word2vec
from src.inputs.readLiarFile import *

flags = tf.app.flags

flags.DEFINE_integer("word_dim", 300, "word embedding size")
flags.DEFINE_integer("num_epochs", 10, "number of epochs")
flags.DEFINE_integer("batch_size", 6, "batch size")

flags.DEFINE_boolean('adv', True, 'set True to adv training')
flags.DEFINE_boolean('test', False, 'set True to test')
flags.DEFINE_boolean('build_data', False, 'set True to generate data')

FLAGS = tf.app.flags.FLAGS

def batch_train(sess, m_train, batch, is_valid):
  best_acc, best_step = 0., 0
  start_time = time.time()
  orig_begin_time = start_time
  all_loss, all_acc = 0., 0.
  batch_size = len(m_train.tensors)
  for i in range(batch_size):
    acc, loss = m_train.tensors[i]
    train_op = m_train.train_ops[i]
    train_fetch = [train_op, loss, acc]
    _, loss, acc = sess.run(train_fetch)
    all_loss += loss
    all_acc += acc

  print('acc_num: %d' % (all_acc))
  all_loss /= batch_size
  all_acc /= batch_size
  print("batch %d all_loss %.2f all_acc %.2f" %
        (batch, all_loss, all_acc))
  if is_valid == False:
    if all_acc > 0.3:
      return 1
    else:
      return 0

def main(_):
  with tf.Graph().as_default():
    train_dir = '../data/liar_train_std.tsv'
    task_labels, labels, sentences = getAllData(train_dir)
    batch_size = FLAGS.batch_size
    batch_num = int(len(labels)/batch_size)
    test_dir = '../data/liar_test_std.tsv'
    test_task_labels, test_labels, test_sentences = getAllData(test_dir)

    for epoch in range(FLAGS.num_epochs):
      for batch in range(batch_num):
        tn_task_labs, tn_labs, tn_sens \
          = task_labels[batch*batch_size: (batch+1)*batch_size], \
            labels[batch*batch_size: (batch+1)*batch_size], sentences[batch*batch_size: (batch+1)*batch_size]
        m_train = mtl_model.build_train_valid_model(
          tn_task_labs, tn_labs, tn_sens, FLAGS.adv, FLAGS.test)
        # m_valid = mtl_model.build_train_valid_model(
        #   test_task_labels, test_labels, test_sentences, FLAGS.adv, FLAGS.test)
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
          sess.run(init_g)
          sess.run(init_l)
          # print('='*80)
          is_valid = batch_train(sess, m_train, batch, is_valid = False)
          # if is_valid == 1:
          #   print('valid...')
          #   batch_train(sess, m_valid, batch = 10000, is_valid = True)
          pass

def getAllData(dir):
  article_vec, creData, labels = getLiarArticLabels(dir=dir)
  task_labels = []
  for i in range(len(labels)):
    if labels[i] == [[0, 0, 0, 0, 0, 1]] or labels[i] == [[0, 0, 0, 1, 0, 0]] or labels[i] == [[0, 0, 1, 0, 0, 0]]:
      task_labels.append([1, 0])
    else:
      task_labels.append([0, 1])
  # print(len(article_vec[0][0]))
  return task_labels, labels, article_vec

if __name__ == '__main__':
  tf.app.run()
