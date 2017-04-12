import tensorflow as tf
from model_base import Model, fully_connected_with_dropout
from tf_decorators import define_scope
from util import roll_batch, unroll_batch
from rnn_encoders import bi_rnn


class Alignment(Model):
    """
    Alignment model without RNN.
    """
    def __init__(self, config, encoding_size, alignment_size):
        Model.__init__(self, config)
        self.name = 'alignment'
        self.encoding_size = encoding_size
        self.alignment_size = alignment_size
        self.premises_encoding
        self.hypotheses_encoding
        self.align
        self.compare
        self.aggregate
        self.logits
        self.loss
        self.optimize
        self.accuracy
        self.predicted_labels
        self.confidences
        self.correct_predictions

    @define_scope
    def premises_encoding(self):
        return self.X.premises

    @define_scope
    def hypotheses_encoding(self):
        return self.X.hypotheses

    @define_scope
    def align(self):
        premises = self.premises_encoding
        hypotheses = self.hypotheses_encoding
        premises_shape = tf.shape(premises)
        hypotheses_shape = tf.shape(hypotheses)
        W_F = tf.Variable(initial_value=tf.random_uniform(shape=[self.encoding_size,
                                                                 self.alignment_size],
                                                          minval=-1.0,
                                                          maxval=1.0,
                                                          dtype=tf.float64),
                          name='W_F')
        premises_unrolled = unroll_batch(premises)                         # [batch_size * max_length_p, encoding_size]
        hypotheses_unrolled = unroll_batch(hypotheses)                     # [batch_size * max_length_h, encoding_size]
        F_p = tf.tanh(tf.matmul(premises_unrolled, W_F), name='F_p')       # [batch_size * max_length_p, align_size]
        F_h = tf.tanh(tf.matmul(hypotheses_unrolled, W_F), name='F_h')     # [batch_size * max_length_h, align_size]
        F_p_rolled = roll_batch(F_p, [premises_shape[0],
                                      premises_shape[1],
                                      self.alignment_size])                # [batch_size, max_length_p, align_size]
        F_h_rolled = roll_batch(F_h, [hypotheses_shape[0],
                                      hypotheses_shape[1],
                                      self.alignment_size])                # [batch_size, max_length_h, align_size]
        eijs = tf.matmul(F_p_rolled,
                         tf.transpose(F_h_rolled,
                                      perm=[0, 2, 1]),
                         name='eijs')                                      # [batch_size, max_length_p, max_length_h]
        eijs_softmaxed = tf.nn.softmax(eijs)                               # [batch_size, max_length_p, max_length_h]
        betas = tf.matmul(eijs_softmaxed, hypotheses)                      # [batch_size, max_length_p, align_size]
        alphas = tf.matmul(tf.transpose(eijs_softmaxed,
                                        perm=[0, 2, 1]),
                           premises)                                       # [batch_size, max_length_h, align_size]
        return betas, alphas

    @define_scope
    def compare(self):
        betas, alphas = self.align
        W_G = tf.Variable(initial_value=tf.random_uniform(shape=[2 * self.encoding_size + self.alignment_size,
                                                                 self.config.ff_size],
                                                          minval=-1.0,
                                                          maxval=1.0,
                                                          dtype=tf.float64),
                          name='W_G')
        V1_in = tf.concat([self.X.premises, betas], axis=2)               # [batch_size, max_length, 2 * encoding_size]
        V2_in = tf.concat([self.X.hypotheses, alphas], axis=2)            # [batch_size, max_length, 2 * encoding_size]
        V1_in_dropped = tf.nn.dropout(V1_in, self.config.p_keep_input)    # [batch_size, max_length, 2 * encoding_size]
        V2_in_dropped = tf.nn.dropout(V2_in, self.config.p_keep_input)    # [batch_size, max_length, 2 * encoding_size]
        V1_in_unrolled = unroll_batch(V1_in_dropped)                      # [batch_size * max_length, 2 * encoding_size]
        V2_in_unrolled = unroll_batch(V2_in_dropped)                      # [batch_size * max_length, 2 * encoding_size]
        V1_unrolled = tf.tanh(tf.matmul(V1_in_unrolled, W_G))             # [batch_size * max_length, ff_size]
        V2_unrolled = tf.tanh(tf.matmul(V2_in_unrolled, W_G))             # [batch_size * max_length, ff_size]
        premises_shape = tf.shape(self.X.premises)
        hypotheses_shape = tf.shape(self.X.hypotheses)
        V1 = roll_batch(V1_unrolled, [premises_shape[0],
                                      premises_shape[1],
                                      self.config.ff_size])
        V2 = roll_batch(V2_unrolled, [hypotheses_shape[0],
                                      hypotheses_shape[1],
                                      self.config.ff_size])
        return V1, V2

    @define_scope
    def aggregate(self):
        V1, V2 = self.compare
        v1 = tf.reduce_sum(V1, axis=1)
        v2 = tf.reduce_sum(V2, axis=1)
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

    @define_scope
    def loss(self):
        cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y,
                                                                              logits=self.logits,
                                                                              name='softmax_cross_entropy'))
        penalty_term = tf.multiply(tf.cast(self.config.lamda, tf.float64),
                                   sum([tf.nn.l2_loss(w) for w in self._all_weights()]),
                                   name='penalty_term')
        return tf.add(cross_entropy, penalty_term, name='loss')

    @define_scope
    def accuracy(self):
        correct_predictions = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float64))
        return accuracy


class BiRNNAlignment(Alignment):
    def __init__(self, config, encoding_size, alignment_size):
        Alignment.__init__(self, config, encoding_size, alignment_size)
        self.name = 'bi_rnn_alignment'

    @define_scope
    def premises_encoding(self):
        _premises_encoding = bi_rnn(self.X.premises,
                                    self.config.rnn_size,
                                    'premise_bi_rnn',
                                    self.config.p_keep_rnn)               # [batch_size, max_length, rnn_size]
        concatenated_encoding = tf.concat(_premises_encoding[0], 2)
        return concatenated_encoding                                      # [batch_size, max_length, 2 * rnn_size]

    @define_scope
    def hypotheses_encoding(self):
        _hypotheses_encodings = bi_rnn(self.X.hypotheses,
                                       self.config.rnn_size,
                                       'hypothesis_bi_rnn',
                                       self.config.p_keep_rnn)            # [batch_size, max_length, rnn_size]
        concatenated_encoding = tf.concat(_hypotheses_encodings[0], 2)
        return concatenated_encoding                                      # [batch_size, max_length, 2 * rnn_size]
