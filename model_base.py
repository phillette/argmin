import tensorflow as tf
from batching import LONGEST_SENTENCE_SNLI
from tf_decorators import define_scope
from util import clip_gradients


class X:
    def __init__(self, premises, hypotheses):
        self.premises = premises
        self.hypotheses = hypotheses


def augment_factor(probability, in_training):
    if in_training:
        return 1.0
    else:
        return probability


def data_placeholders(time_steps, word_embed_size):
    premises = tf.placeholder(tf.float64,
                              [None,
                               time_steps,
                               word_embed_size],
                              name='premises')
    hypotheses = tf.placeholder(tf.float64,
                                [None,
                                 time_steps,
                                 word_embed_size],
                                name='hypotheses')
    Y = tf.placeholder(tf.float64,
                       [None, 3],
                       name='y')
    return X(premises, hypotheses), Y


def p_drop(probability, in_training):
    if in_training:
        return probability
    else:
        return 1.0


def fully_connected_with_dropout(inputs, num_outputs, activation_fn, p_keep):
    fully_connected = tf.contrib.layers.fully_connected(inputs, num_outputs, activation_fn)
    dropped_out = tf.nn.dropout(fully_connected, p_keep)
    return dropped_out


class Config:
    def __init__(self,
                 word_embed_size=300,
                 learning_rate=1e-3,
                 time_steps=LONGEST_SENTENCE_SNLI,
                 grad_clip_norm=5.0,
                 hidden_size=100,
                 rnn_size=300,
                 ff_size=200,
                 lamda=0.9,
                 p_keep_input=0.8,
                 p_keep_rnn=0.5,
                 p_keep_ff=0.5):
        self.word_embed_size = word_embed_size
        self.learning_rate = learning_rate
        self.time_steps = time_steps
        self.grad_clip_norm = grad_clip_norm
        self.hidden_size = hidden_size
        self.rnn_size = rnn_size
        self.ff_size = ff_size
        self.lamda = lamda
        self.p_keep_input = p_keep_input
        self.p_keep_rnn = p_keep_rnn
        self.p_keep_ff = p_keep_ff


class Model:
    def __init__(self, config):
        self.config = config
        self.global_step = tf.Variable(0,
                                       dtype=tf.int32,
                                       trainable=False,
                                       name='global_step')
        self.in_training = False
        self._data

    def _augment_factor(self, probability):
        return augment_factor(probability, self.in_training)

    def _p_drop(self, probability):
        return p_drop(probability, self.in_training)

    def _augment_weights(self, scope):
        pass

    @define_scope()
    def accuracy_train(self):
        return self.accuracy

    @define_scope()
    def accuracy(self):
        correct_predictions = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float64))
        return accuracy

    @define_scope('data')
    def _data(self):
        self.X, self.Y = data_placeholders(self.config.time_steps,
                                           self.config.word_embed_size)
        return self.X, self.Y

    @define_scope
    def loss(self):
        return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.y,
                                                                     logits=self.logits,
                                                                     name='loss'))

    @define_scope
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.loss, self._weights)
        clipped_grads_and_vars = clip_gradients(grads_and_vars)
        return optimizer.apply_gradients(clipped_grads_and_vars)

    def _weights(self):
        return [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('weights:0')]
