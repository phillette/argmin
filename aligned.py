import tensorflow as tf
from model_base import Model, fully_connected_with_dropout
from tf_decorators import define_scope
from util import length


class Alignment(Model):
    """
    Alignment model without RNN.
    """
    def __init__(self, config, align_hidden_units):
        Model.__init__(self, config)
        self.name = 'alignment'
        self.align_hidden_units = align_hidden_units

    @define_scope
    def align(self):
        W_F = tf.Variable(tf.float64,
                          [self.config.word_embed_size,
                           self.align_hidden_units],
                          name='W_F')
        F_p = tf.tanh(tf.matmul(self.X.premises, W_F), name='F_p')
        F_h = tf.tanh(tf.matmul(self.X.hypotheses, W_F), name='F_h')
        eijs = tf.matmul(F_p, tf.transpose(F_h), name='eijs')
        betas = tf.matmul(tf.nn.softmax(eijs),  # might double check the orientation is correct
                          self.X.hypotheses)
        alphas = tf.matmul(tf.nn.softmax(tf.transpose(eijs)),  # check orientation
                           self.X.premises)
        return betas, alphas

    @define_scope
    def compare(self):
        betas, alphas = self.align
        W_G = tf.Variable(tf.float64,
                          [2 * self.config.word_embed_size,
                           self.config.ff_size],
                          name='W_G')
        premise_length = tf.shape(self.X.premises, 2)
        hypothesis_length = tf.shape(self.X.hypotheses, 2)
        return None

    @define_scope
    def aggregate(self):
        V1, V2 = self.compare
        v1 = tf.reduce_sum(V1, axis=2)
        v2 = tf.reduce_sum(V2, axis=2)
        concatenated = tf.concat([v1, v2], axis=1)
        return concatenated

    @define_scope
    def logits(self):
        concatenated = self.aggregate
        dropped_input = tf.nn.dropout(concatenated, self.config.p_keep_input)
        a1 = fully_connected_with_dropout(dropped_input,
                                          self.config.ff_size,
                                          tf.tanh,
                                          self.config.p_keep_ff)
        a2 = fully_connected_with_dropout(a1,
                                          self.config.ff_size,
                                          tf.tanh,
                                          self.config.p_keep_ff)
        a3 = tf.contrib.layers.fully_connected(a2, 3, None)
        return a3
