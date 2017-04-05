import tensorflow as tf
from batching import LONGEST_SENTENCE_SNLI
from tf_decorators import define_scope
from util import clip_gradients


def fully_connected_with_dropout(inputs, num_outputs, activation_fn, p_keep):
    fully_connected = tf.contrib.layers.fully_connected(inputs, num_outputs, activation_fn)
    dropped_out = tf.nn.dropout(fully_connected, p_keep)
    return dropped_out


class Config:
    def __init__(self,
                 word_embed_length=300,
                 learning_rate=1e-3,
                 time_steps=LONGEST_SENTENCE_SNLI,
                 grad_norm=5.0,
                 hidden_size=100,
                 rnn_size=300,
                 ff_size=200,
                 lamda=0.9,
                 p_keep_input=0.8,
                 p_keep_rnn=0.5,
                 p_keep_ff=0.5):
        self.word_embed_length = word_embed_length
        self.learning_rate = learning_rate
        self.time_steps = time_steps
        self.grad_norm = grad_norm
        self.hidden_size = hidden_size
        self.rnn_size = rnn_size
        self.ff_size = ff_size
        self.lamda = lamda
        self.p_keep_input = p_keep_input
        self.p_keep_rnn = p_keep_rnn
        self.p_keep_ff = p_keep_ff


class Model:
    def __init__(self, config):
        self.word_embed_length = config.word_embed_length
        self.learning_rate = config.learning_rate
        self.lamda = config.lamda
        self.time_steps = config.time_steps
        self.grad_norm = config.grad_norm
        self.hidden_size = config.hidden_size
        self.rnn_size = config.rnn_size
        self.ff_size = config.ff_size
        self.p_keep_input = config.p_keep_input
        self.p_keep_rnn = config.p_keep_rnn
        self.p_keep_ff = config.p_keep_ff
        self.global_step = tf.Variable(0,
                                       dtype=tf.int32,
                                       trainable=False,
                                       name='global_step')
        self._data
        self.logits
        self.loss
        self.optimize
        self.accuracy_train
        self.accuracy

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
        self.premises = tf.placeholder(tf.float64,
                                       [None,
                                        self.time_steps,
                                        self.word_embed_length],
                                       name='premises')
        self.hypotheses = tf.placeholder(tf.float64,
                                         [None,
                                          self.time_steps,
                                          self.word_embed_length],
                                         name='hypotheses')
        self.y = tf.placeholder(tf.float64,
                                [None, 3],
                                name='y')
        return self.premises, self.hypotheses, self.y

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
