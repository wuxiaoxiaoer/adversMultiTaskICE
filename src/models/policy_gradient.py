
import numpy as np
import tensorflow as tf
import gym
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print (mnist.train.images.shape)

input_size = 28
timestep_size = 28
hidden_size = 256
class_num = 10
batch_size = 128

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
_inputs = tf.placeholder(tf.float32, [None, 784])
inputs = tf.reshape(_inputs, [-1, 28, 28])
y = tf.placeholder(tf.float32, [None, class_num])

W = tf.Variable(tf.truncated_normal(shape=[hidden_size, class_num]))
b = tf.Variable(tf.zeros(shape=[class_num]))

np.random.seed(1)
tf.set_random_seed(1)

class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()

        self.sess = tf.Session()

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")

        layer = tf.layers.dense(
            inputs=self.tf_obs,
            units=10,
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )

        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )
        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')

        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
            # or in this way:
            # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observ):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observ})
        actions = np.zeros([batch_size,1])
        for o in range(len(prob_weights)):
            action0 = np.random.choice(range(prob_weights[o].shape[0]), p=prob_weights[o].ravel())
            actions[o][0] = action0
        return actions

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn2(self, observ, y, steps):
        logits = tf.nn.softmax(tf.matmul(observ, W) + b)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
        with tf.variable_scope('train_accuracy', reuse=tf.AUTO_REUSE):
            train_op = tf.train.AdamOptimizer(self.lr).minimize(cross_entropy)
        max_logits = tf.argmax(logits, 1)
        max_y = tf.argmax(y, 1)
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        with tf.Session() as sess:
            sess = tf.Session()
            sess.run(tf.initialize_all_variables())
            cross_entropy_ = sess.run(cross_entropy)
        rewards = np.log(cross_entropy_) + 0.5 * (steps/input_size)

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []  # empty episode data
        return train_op, rewards

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add
        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

def observ_step(action, sentences, adjusted_sentence, timestep):
    reward = 0.0
    for o in range(len(action)):
        if action[o] == 1:
            word = sentences[o][timestep]
            reward = 1.0
        if action[o] == 0:
            word = [0]*input_size
            reward = 0.0
        np.r_(adjusted_sentence[o], word)
    done = False
    with tf.variable_scope('lstm', reuse=tf.AUTO_REUSE):
        basic_cell = rnn.BasicLSTMCell(hidden_size)
        outputs, states = tf.nn.dynamic_rnn(basic_cell, adjusted_sentence, dtype=tf.float32)
    observ = outputs[:, -1, :]
    with tf.Session() as sess:
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        obs = sess.run(observ)
    if timestep >=27:
        done = True
    return obs, reward, done

env = gym.make('CartPole-v0').unwrapped
rl = PolicyGradient(
    n_actions=2,
    n_features=hidden_size,
    learning_rate=0.02,
    reward_decay=0.99
)

for i in range(10):
    _batch_size = 128
    batch = mnist.train.next_batch(_batch_size)
    sentences = sess.run(inputs, feed_dict={_inputs : batch[0]})
    ys = sess.run(y, feed_dict={y:batch[1]})
    flag = 0
    for sentence in sentences:
        for episode in range(500):
            _observ = sentences[:, -1, :]
            print(np.shape(_observ))
            print(len(_observ))
            print(len(_observ[0]))
            adjusted_sentence = np.reshape(_observ, [len(_observ), 1, len(_observ[0])])
            observ = np.zeros([batch_size, hidden_size])
            for o in range(len(_observ)):
                for i in range(hidden_size):
                    if i < len(_observ[o]):
                        observ[o][i] = _observ[o][i]
                    else:
                        observ[o][i] = 0.0
            steps = 0
            timestep = 0
            while True:
                action = rl.choose_action(observ)
                observation_, step, done = observ_step(action, sentences, adjusted_sentence, timestep)
                steps += step
                rl.store_transition(observ, action, steps)
                if done:
                    print('episode: ', episode, 'steps: ', steps)
                    ys_one = np.reshape(ys[flag], (1, len(ys[flag])))
                    train_op, rewards = rl.learn2(observation_, ys_one, steps)
                    observ = []
                    break
                observ = observation_
                timestep += 1
        flag += 1