import tensorflow as tf
import model_base
import decorators
import util
import rnn_encoders


# control randomization for reproducibility
tf.set_random_seed(1984)


class AlignmentOld(model_base.Model):
    """
    Alignment model without RNN.
    """
    def __init__(self, config, alignment_size):
        model_base.Model.__init__(self, config)
        self.name = 'alignment'
        self.alignment_size = alignment_size
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

    @decorators.define_scope
    def align(self):
        W_F = tf.Variable(initial_value=tf.random_uniform(shape=[self.config.word_embed_size,
                                                                 self.alignment_size],
                                                          minval=-1.0,
                                                          maxval=1.0,
                                                          dtype=tf.float64),
                          name='W_F')
        premises_shape = tf.shape(self.X.premises)
        hypotheses_shape = tf.shape(self.X.hypotheses)
        premises_unrolled = util.unroll_batch(self.X.premises)                     # [batch_size * max_length_p, embed_size]
        hypotheses_unrolled = util.unroll_batch(self.X.hypotheses)                 # [batch_size * max_length_h, embed_size]
        F_p = tf.tanh(tf.matmul(premises_unrolled, W_F), name='F_p')          # [batch_size * max_length_p, align_size]
        F_h = tf.tanh(tf.matmul(hypotheses_unrolled, W_F), name='F_h')        # [batch_size * max_length_h, align_size]
        F_p_rolled = util.roll_batch(F_p, [premises_shape[0],
                                      premises_shape[1],
                                      self.alignment_size])                   # [batch_size, max_length_p, align_size]
        F_h_rolled = util.roll_batch(F_h, [hypotheses_shape[0],
                                      hypotheses_shape[1],
                                      self.alignment_size])                   # [batch_size, max_length_h, align_size]
        eijs = tf.matmul(F_p_rolled,
                         tf.transpose(F_h_rolled,
                                      perm=[0, 2, 1]),
                         name='eijs')                                         # [batch_size, max_length_p, max_length_h]
        eijs_softmaxed = tf.nn.softmax(eijs)                                  # [batch_size, max_length_p, max_length_h]
        betas = tf.matmul(eijs_softmaxed,
                          self.X.hypotheses)                                  # [batch_size, max_length_p, embed_size]
        alphas = tf.matmul(tf.transpose(eijs_softmaxed,
                                        perm=[0, 2, 1]),
                           self.X.premises)                                   # [batch_size, max_length_h, embed_size]
        return betas, alphas

    @decorators.define_scope
    def compare(self):
        betas, alphas = self.align
        W_G = tf.Variable(initial_value=tf.random_uniform(shape=[2 * self.config.word_embed_size,
                                                                 self.config.ff_size],
                                                          minval=-1.0,
                                                          maxval=1.0,
                                                          dtype=tf.float64),
                          name='W_F')
        V1_in = tf.concat([self.X.premises, betas], axis=2)                  # [batch_size, max_length, 2 * embed_size]
        V2_in = tf.concat([self.X.hypotheses, alphas], axis=2)               # [batch_size, max_length, 2 * embed_size]
        V1_in_dropped = tf.nn.dropout(V1_in, self.config.p_keep_input)       # [batch_size, max_length, 2 * embed_size]
        V2_in_dropped = tf.nn.dropout(V2_in, self.config.p_keep_input)       # [batch_size, max_length, 2 * embed_size]
        V1_in_unrolled = util.unroll_batch(V1_in_dropped)                         # [batch_size * max_length, 2 * embed_size]
        V2_in_unrolled = util.unroll_batch(V2_in_dropped)                         # [batch_size * max_length, 2 * embed_size]
        V1_unrolled = tf.tanh(tf.matmul(V1_in_unrolled, W_G))                # [batch_size * max_length, ff_size]
        V2_unrolled = tf.tanh(tf.matmul(V2_in_unrolled, W_G))                # [batch_size * max_length, ff_size]
        premises_shape = tf.shape(self.X.premises)
        hypotheses_shape = tf.shape(self.X.hypotheses)
        V1 = util.roll_batch(V1_unrolled, [premises_shape[0],
                                      premises_shape[1],
                                      self.config.ff_size])
        V2 = util.roll_batch(V2_unrolled, [hypotheses_shape[0],
                                      hypotheses_shape[1],
                                      self.config.ff_size])
        return V1, V2

    @decorators.define_scope
    def aggregate(self):
        V1, V2 = self.compare
        v1 = tf.reduce_sum(V1, axis=1)
        v2 = tf.reduce_sum(V2, axis=1)
        concatenated = tf.concat([v1, v2], axis=1)
        return concatenated

    @decorators.define_scope
    def logits(self):
        concatenated = self.aggregate
        dropped_input = tf.nn.dropout(concatenated, self.config.p_keep_input)
        a1 = model_base.fully_connected_with_dropout(dropped_input,
                                                     self.config.ff_size,
                                                     tf.tanh,
                                                     self.config.p_keep_ff)
        a2 = model_base.fully_connected_with_dropout(a1,
                                                     self.config.ff_size,
                                                     tf.tanh,
                                                     self.config.p_keep_ff)
        a3 = tf.contrib.layers.fully_connected(a2, 3, None)
        return a3

    @decorators.define_scope
    def loss(self):
        cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y,
                                                                              logits=self.logits,
                                                                              name='softmax_cross_entropy'))
        penalty_term = tf.multiply(tf.cast(self.config.lamda, tf.float64),
                                   sum([tf.nn.l2_loss(w) for w in self._all_weights()]),
                                   name='penalty_term')
        return tf.add(cross_entropy, penalty_term, name='loss')

    @decorators.define_scope
    def accuracy(self):
        correct_predictions = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float64))
        return accuracy


class Alignment(model_base.Model):
    """
    Alignment model without RNN.
    """
    def __init__(self, config, encoding_size, alignment_size, activation=tf.tanh):
        model_base.Model.__init__(self, config)
        self.name = 'alignment'
        self.encoding_size = encoding_size
        self.alignment_size = alignment_size
        self.activation = activation
        self.parameters
        self.biases
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
        self.summaries

    @decorators.define_scope
    def biases(self):
        self.b_F_p = tf.Variable(initial_value=tf.random_uniform(shape=[1, self.alignment_size],
                                                                 minval=-0.1,
                                                                 maxval=0.1,
                                                                 dtype=tf.float64),
                                 name='biases_F_p')
        self.b_F_h = tf.Variable(initial_value=tf.random_uniform(shape=[1, self.alignment_size],
                                                                 minval=-0.1,
                                                                 maxval=0.1,
                                                                 dtype=tf.float64),
                                 name='biases_F_h')
        self.b_G = tf.Variable(initial_value=tf.random_uniform(shape=[1, self.config.ff_size],
                                                               minval=-0.1,
                                                               maxval=0.1,
                                                               dtype=tf.float64),
                               name='biases_G')
        return self.b_F_p, self.b_F_h, self.b_G

    @decorators.define_scope
    def parameters(self):
        self.W_F = tf.Variable(initial_value=tf.random_uniform(shape=[self.encoding_size,
                                                                      self.alignment_size],
                                                               minval=-0.1,
                                                               maxval=0.1,
                                                               dtype=tf.float64),
                               name='Weights_F')
        self.W_G = tf.Variable(initial_value=tf.random_uniform(shape=[2 * self.encoding_size,
                                                                      self.config.ff_size],
                                                               minval=-0.1,
                                                               maxval=0.1,
                                                               dtype=tf.float64),
                               name='Weights_G')
        return self.W_F, self.W_G

    @decorators.define_scope
    def premises_encoding(self):
        return self.X.premises

    @decorators.define_scope
    def hypotheses_encoding(self):
        return self.X.hypotheses

    @decorators.define_scope
    def align(self):
        premises = self.premises_encoding
        hypotheses = self.hypotheses_encoding
        premises_shape = tf.shape(premises)
        hypotheses_shape = tf.shape(hypotheses)
        premises_unrolled = util.unroll_batch(premises)                         # [batch_size * max_length_p, encoding_size]
        premises_unrolled.set_shape([None, self.encoding_size])
        hypotheses_unrolled = util.unroll_batch(hypotheses)                     # [batch_size * max_length_h, encoding_size]
        hypotheses_unrolled.set_shape([None, self.encoding_size])
        #z_F_p = tf.add(tf.matmul(premises_unrolled,
        #                         self.W_F),
        #               self.b_F)
        #z_F_h = tf.add(tf.matmul(hypotheses_unrolled,
        #                         self.W_F),
        #               self.b_F)
        #F_p = self.activation(z_F_p)                                       # [batch_size * max_length_p, align_size]
        #F_h = self.activation(z_F_h)                                       # [batch_size * max_length_h, align_size]
        F_p = model_base.fully_connected_with_dropout(inputs=premises_unrolled,
                                                      num_outputs=self.alignment_size,
                                                      activation_fn=self.activation,
                                                      p_keep=self.config.p_keep_ff)
        F_h = model_base.fully_connected_with_dropout(inputs=hypotheses_unrolled,
                                                      num_outputs=self.alignment_size,
                                                      activation_fn=self.activation,
                                                      p_keep=self.config.p_keep_ff)
        F_p_rolled = util.roll_batch(F_p, [premises_shape[0],
                                      premises_shape[1],
                                      self.alignment_size])                # [batch_size, max_length_p, align_size]
        F_h_rolled = util.roll_batch(F_h, [hypotheses_shape[0],
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

    @decorators.define_scope
    def compare(self):
        betas, alphas = self.align
        V1_in = tf.concat([self.X.premises, betas], axis=2)               # [batch_size, max_length, 2 * encoding_size]
        V2_in = tf.concat([self.X.hypotheses, alphas], axis=2)            # [batch_size, max_length, 2 * encoding_size]
        V1_in_dropped = tf.nn.dropout(V1_in, self.config.p_keep_input)    # [batch_size, max_length, 2 * encoding_size]
        V2_in_dropped = tf.nn.dropout(V2_in, self.config.p_keep_input)    # [batch_size, max_length, 2 * encoding_size]
        V1_in_unrolled = util.unroll_batch(V1_in_dropped)                      # [batch_size * max_length, 2 * encoding_size]
        V1_in_unrolled.set_shape([None, 2 * self.encoding_size])
        V2_in_unrolled = util.unroll_batch(V2_in_dropped)                      # [batch_size * max_length, 2 * encoding_size]
        V2_in_unrolled.set_shape([None, 2 * self.encoding_size])
        #z_G_v1 = tf.add(tf.matmul(V1_in_unrolled,
        #                          self.W_G),
        #                self.b_G)
        #z_G_v2 = tf.add(tf.matmul(V2_in_unrolled,
        #                          self.W_G),
        #                self.b_G)
        #V1_unrolled = self.activation(z_G_v1)                             # [batch_size * max_length, ff_size]
        #V2_unrolled = self.activation(z_G_v2)                             # [batch_size * max_length, ff_size]
        V1_unrolled = model_base.fully_connected_with_dropout(inputs=V1_in_unrolled,
                                                              num_outputs=self.config.ff_size,
                                                              activation_fn=self.activation,
                                                              p_keep=self.config.p_keep_ff)
        V2_unrolled = model_base.fully_connected_with_dropout(inputs=V2_in_unrolled,
                                                              num_outputs=self.config.ff_size,
                                                              activation_fn=self.activation,
                                                              p_keep=self.config.p_keep_ff)
        premises_shape = tf.shape(self.X.premises)
        hypotheses_shape = tf.shape(self.X.hypotheses)
        V1 = util.roll_batch(V1_unrolled, [premises_shape[0],
                                      premises_shape[1],
                                      self.config.ff_size])
        V2 = util.roll_batch(V2_unrolled, [hypotheses_shape[0],
                                      hypotheses_shape[1],
                                      self.config.ff_size])
        return V1, V2

    @decorators.define_scope
    def aggregate(self):
        V1, V2 = self.compare
        v1 = tf.reduce_sum(V1, axis=1)
        v2 = tf.reduce_sum(V2, axis=1)
        concatenated = tf.concat([v1, v2], axis=1)
        return concatenated

    @decorators.define_scope
    def logits(self):
        concatenated = self.aggregate
        dropped_input = tf.nn.dropout(concatenated, self.config.p_keep_input)
        a1 = model_base.fully_connected_with_dropout(dropped_input,
                                                     self.config.ff_size,
                                                     tf.nn.relu,
                                                     self.config.p_keep_ff)
        a2 = model_base.fully_connected_with_dropout(a1,
                                                     self.config.ff_size,
                                                     tf.nn.relu,
                                                     self.config.p_keep_ff)
        a3 = tf.contrib.layers.fully_connected(a2, 3, None)
        return a3

    @decorators.define_scope
    def loss(self):
        cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y,
                                                                              logits=self.logits,
                                                                              name='softmax_cross_entropy'))
        penalty_term = tf.multiply(tf.cast(self.config.lamda, tf.float64),
                                   sum([tf.nn.l2_loss(w) for w in self._all_weights()]),
                                   name='penalty_term')
        return tf.add(cross_entropy, penalty_term, name='loss')

    @decorators.define_scope
    def accuracy(self):
        correct_predictions = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float64))
        return accuracy


class BiRNNAlignment(Alignment):
    def __init__(self, config, encoding_size, alignment_size):
        Alignment.__init__(self, config, encoding_size, alignment_size)
        self.name = 'bi_rnn_alignment'

    @decorators.define_scope
    def parameters(self):
        self.W_F = tf.Variable(initial_value=tf.random_uniform(shape=[self.encoding_size,
                                                                      self.alignment_size],
                                                               minval=-1.0,
                                                               maxval=1.0,
                                                               dtype=tf.float64),
                               name='W_F')
        self.W_G = tf.Variable(initial_value=tf.random_uniform(shape=[2 * self.encoding_size + self.alignment_size,
                                                                      self.config.ff_size],
                                                               minval=-1.0,
                                                               maxval=1.0,
                                                               dtype=tf.float64),
                               name='W_G')
        return self.W_F, self.W_G

    @decorators.define_scope
    def premises_encoding(self):
        _premises_encoding = rnn_encoders.bi_rnn(self.X.premises,
                                                 self.config.rnn_size,
                                                 'premise_bi_rnn',
                                                 self.config.p_keep_rnn)            # [batch_size, max_length, rnn_size]
        concatenated_encoding = tf.concat(_premises_encoding[0], 2)
        return concatenated_encoding                                            # [batch_size, max_length, 2 * rnn_size]

    @decorators.define_scope
    def hypotheses_encoding(self):
        _hypotheses_encodings = rnn_encoders.bi_rnn(self.X.hypotheses,
                                                    self.config.rnn_size,
                                                    'hypothesis_bi_rnn',
                                                    self.config.p_keep_rnn)         # [batch_size, max_length, rnn_size]
        concatenated_encoding = tf.concat(_hypotheses_encodings[0], 2)
        return concatenated_encoding                                            # [batch_size, max_length, 2 * rnn_size]


class AlignmentParikh(model_base.Model):
    def __init__(self, config, encoding_size=300, alignment_size=200, projection_size=200, activation=tf.nn.relu):
        model_base.Model.__init__(self, config)
        self.name = 'alignment'
        self.encoding_size = encoding_size
        self.alignment_size = alignment_size
        self.projection_size = projection_size
        self.activation = activation
        self.premises_encoding
        self.hypotheses_encoding
        self.project
        self.align
        self.compare
        self.aggregate
        self.logits
        self.loss
        self.optimize
        self.predicted_labels
        self.correct_predictions
        self.accuracy
        self.confidences
        self.summaries

    @decorators.define_scope
    def premises_encoding(self):
        return self.X.premises    # [batch_size, MAX_SENTENCE_LENGTH_SNLI, embed_size]

    @decorators.define_scope
    def hypotheses_encoding(self):
        return self.X.hypotheses  # [batch_size, MAX_SENTENCE_LENGTH_SNLI, embed_size]

    @decorators.define_scope
    def project(self):
        concatenated = util.concat(self.premises_encoding,
                                   self.hypotheses_encoding)            # [2 * batch_size, len_longest_sent, embed_size]
        projected = model_base.fully_connected_with_dropout(concatenated,
                                                            self.projection_size,
                                                            self.activation,
                                                            self.config.p_keep_ff)
        return projected

    @decorators.define_scope
    def align(self):
        Fs = model_base.fully_connected_with_dropout(inputs=self.project,
                                                     num_outputs=self.alignment_size,
                                                     activation_fn=self.activation,
                                                     p_keep=self.config.p_keep_ff)
        F_premises, F_hypotheses = util.split_after_concat(Fs,                # [batch_size, time_steps, alignment_size]
                                                           tf.shape(self.X.premises)[0])
        eijs = tf.matmul(F_premises,
                         tf.transpose(F_hypotheses,
                                      perm=[0, 2, 1]),
                         name='eijs')                         # [batch_size, time_steps, time_steps]
        eijs_softmaxed = tf.nn.softmax(eijs)                  # [batch_size, time_steps, time_steps]
        betas = tf.matmul(eijs_softmaxed,
                          self.hypotheses_encoding)           # [batch_size, time_steps, alignment_size]
        alphas = tf.matmul(tf.transpose(eijs_softmaxed,
                                        perm=[0, 2, 1]),
                           self.premises_encoding)            # [batch_size, time_steps, alignment_size]
        return betas, alphas

    @decorators.define_scope
    def compare(self):
        betas, alphas = self.align
        concatenated = util.concat(betas, alphas)
        Vs = model_base.fully_connected_with_dropout(inputs=concatenated,
                                                     num_outputs=self.config.ff_size,
                                                     activation_fn=self.activation,
                                                     p_keep=self.config.p_keep_ff)
        V1, V2 = util.split_after_concat(Vs,                                         # [batch_size, time_steps, ff_size]
                                         tf.shape(self.X.premises)[0])
        return V1, V2

    @decorators.define_scope
    def aggregate(self):
        V1, V2 = self.compare                       # [batch_size, time_steps, ff_size]
        v1 = tf.reduce_sum(V1, axis=1)              # [batch_size, 1, ff_size]
        v2 = tf.reduce_sum(V2, axis=1)              # [batch_size, 1, ff_size]
        concatenated = tf.concat([v1, v2], axis=1)  # [batch_size, 2, ff_size]  can't be right...
        concatenated.set_shape([None, 2 * self.config.ff_size])  # because this is 2d
        return concatenated   # [batch_size, 2 * ff_size]

    @decorators.define_scope
    def logits(self):
        a1 = model_base.fully_connected_with_dropout(self.aggregate,
                                                     self.config.ff_size,
                                                     self.activation,
                                                     self.config.p_keep_ff)
        a2 = model_base.fully_connected_with_dropout(a1,
                                                     self.config.ff_size,
                                                     self.activation,
                                                     self.config.p_keep_ff)
        a3 = tf.contrib.layers.fully_connected(a2, 3, None)
        return a3

    @decorators.define_scope
    def loss(self):
        cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y,
                                                                              logits=self.logits,
                                                                              name='softmax_cross_entropy'))
        penalty_term = tf.multiply(tf.cast(self.config.lamda, tf.float64),
                                   sum([tf.nn.l2_loss(w) for w in self._all_weights()]),
                                   name='penalty_term')
        return tf.add(cross_entropy, penalty_term, name='loss')
