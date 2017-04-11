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
        self.data

    @define_scope
    def accuracy(self):
        return tf.reduce_mean(tf.cast(self.correct_predictions, tf.float64))

    @define_scope
    def confidences(self):
        return tf.reduce_max(self.logits, axis=1)

    @define_scope
    def correct_predictions(self):
        return tf.equal(self.predicted_labels, tf.argmax(self.Y, axis=1))

    @define_scope
    def data(self):
        self.X, self.Y = data_placeholders(self.config.time_steps,
                                           self.config.word_embed_size)
        return self.X, self.Y

    @define_scope
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.loss, self._all_weights())
        clipped_grads_and_vars = clip_gradients(grads_and_vars)
        return optimizer.apply_gradients(clipped_grads_and_vars)

    @define_scope
    def predicted_labels(self):
        return tf.argmax(self.logits, axis=1)

    def _weights(self, scope):
        vars = tf.global_variables()
        weights_name = '%s/weights:0' % scope
        if weights_name not in [v.name for v in vars]:
            raise Exception('Could not find weights with name %s' % weights_name)
        return next(v for v in vars if v.name == weights_name)

    def _all_weights(self):
        return [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('weights:0')]
