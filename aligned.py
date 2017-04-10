import tensorflow as tf
from model_base import Model
from tf_decorators import define_scope


class Alignment(Model):
    """
    Alignment model without RNN.
    """
    def __init__(self, config, align_hidden_units):
        Model.__init__(self, config)
        self.name = 'alignment'
        self.align_hidden_units = align_hidden_units

    @define_scope
    def alignments(self):
        W_F = tf.Variable(tf.float64,
                          [self.config.word_embed_size,
                           self.align_hidden_units],
                          name='W_F')
        F_p = tf.tanh(tf.matmul(self.X.premises, W_F), name='F_p')
        F_h = tf.tanh(tf.matmul(self.X.hypotheses, W_F), name='F_h')
        eijs = tf.matmul(F_p, tf.transpose(F_h), name='eijs')
        betas = None
        alphas = None
