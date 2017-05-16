import tensorflow as tf
import model_base
import decorators
import util
import rnn_encoders


# control randomization for reproducibility
tf.set_random_seed(1984)


class Alignment(model_base.Model):
    def __init__(self, config):
        model_base.Model.__init__(self, config)
        self.name = 'alignment'
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
        self.summary

    @decorators.define_scope
    def premises_encoding(self):
        # [batch_size, timesteps, embed_size]
        return self.premises

    @decorators.define_scope
    def hypotheses_encoding(self):
        # [batch_size, timesteps, embed_size]
        return self.hypotheses

    @decorators.define_scope
    def project(self):
        # [2 * batch_size, timesteps, embed_size]
        concatenated = util.concat(
            premises=self.premises_encoding,
            hypotheses=self.hypotheses_encoding)

        # [2 * batch_size, timesteps, hidden_size]
        projected = model_base.fully_connected_with_dropout(
            inputs=concatenated,
            num_outputs=self.hidden_size,
            activation_fn=None,
            p_keep=self.p_keep)

        return projected

    @decorators.define_scope
    def align(self):
        # [2 * batch_size, timesteps, hidden_size]
        Fs1 = model_base.fully_connected_with_dropout(
             inputs=self.project,
             num_outputs=self.hidden_size,
             activation_fn=tf.nn.relu,
             p_keep=self.p_keep)
        Fs2 = model_base.fully_connected_with_dropout(
            inputs=Fs1,
            num_outputs=self.hidden_size,
            activation_fn=tf.nn.relu,
            p_keep=self.p_keep
        )

        # [batch_size, timesteps, hidden_size]
        F_premises, F_hypotheses = util.split_after_concat(
            tensor=Fs2,
            batch_size=self.batch_size)

        # [batch_size, timesteps, timesteps]
        eijs = tf.matmul(F_premises,
                         tf.transpose(F_hypotheses,
                                      perm=[0, 2, 1]),
                         name='eijs')

        # [batch_size, timesteps, timesteps]
        eijs_softmaxed_for_premises = tf.nn.softmax(eijs)
        eijs_softmaxed_for_hypotheses = tf.nn.softmax(
            tf.transpose(
                eijs,
                perm=[0, 2, 1]))

        # [batch_size, timesteps, hidden_size]
        betas = tf.matmul(eijs_softmaxed_for_premises,
                          self.hypotheses_encoding)

        # [batch_size, timesteps, hidden_size]
        alphas = tf.matmul(eijs_softmaxed_for_hypotheses,
                           self.premises_encoding)

        return betas, alphas

    @decorators.define_scope
    def compare(self):
        betas, alphas = self.align

        # [batch_size, timesteps, 2 * hidden_size]
        V1_input = tf.concat([self.premises_encoding,
                              betas],
                             axis=2)
        # [batch_size, timesteps, 2 * hidden_size]
        V2_input = tf.concat([self.hypotheses_encoding,
                              alphas],
                             axis=2)

        # [2 * batch_size, timesteps, 2 * hidden_size]
        ff_input = util.concat(V1_input, V2_input)
        Vs1 = model_base.fully_connected_with_dropout(
             inputs=ff_input,
             num_outputs=self.hidden_size,
             activation_fn=tf.nn.relu,
             p_keep=self.p_keep)
        Vs2 = model_base.fully_connected_with_dropout(
            inputs=Vs1,
            num_outputs=self.hidden_size,
            activation_fn=tf.nn.relu,
            p_keep=self.p_keep
        )

        # [batch_size, timesteps, hidden_size]
        V1, V2 = util.split_after_concat(Vs2, self.batch_size)

        return V1, V2

    @decorators.define_scope
    def aggregate(self):
        # [batch_size, timesteps, hidden_size]
        V1, V2 = self.compare

        # new aggregation method (Chen)
        avg_premises = tf.reduce_mean(V1, axis=1)
        max_premises = tf.reduce_max(V1, axis=1)
        avg_hypotheses = tf.reduce_mean(V2, axis=1)
        max_hypotheses = tf.reduce_max(V2, axis=1)
        concatenated = tf.concat([avg_premises,
                                  max_premises,
                                  avg_hypotheses,
                                  max_hypotheses],
                                 axis=1)
        concatenated.set_shape([None, 4 * self.hidden_size])

        return concatenated

    @decorators.define_scope
    def logits(self):
        a1 = model_base.fully_connected_with_dropout(
            inputs=self.aggregate,
            num_outputs=self.hidden_size,
            activation_fn=tf.nn.relu,
            p_keep=self.p_keep)
        a2 = model_base.fully_connected_with_dropout(
            inputs=a1,
            num_outputs=self.hidden_size,
            activation_fn=tf.nn.relu,
            p_keep=self.p_keep)
        a3 = tf.contrib.layers.fully_connected(a2, 3, None)
        return a3


class AlignmentDeep(Alignment):
    def __init__(self, config):
        Alignment.__init__(self, config)
        self.name = 'alignment_deep'

    @decorators.define_scope
    def align(self):
        # [2 * batch_size, timesteps, hidden_size]
        Fs1 = model_base.fully_connected_with_dropout(
            inputs=self.project,
            num_outputs=self.hidden_size,
            activation_fn=tf.nn.relu,
            p_keep=self.p_keep)
        Fs2 = model_base.fully_connected_with_dropout(
            inputs=Fs1,
            num_outputs=self.hidden_size,
            activation_fn=tf.nn.relu,
            p_keep=self.p_keep
        )
        Fs3 = model_base.fully_connected_with_dropout(
            inputs=Fs2,
            num_outputs=self.hidden_size,
            activation_fn=tf.nn.relu,
            p_keep=self.p_keep
        )

        # [batch_size, timesteps, hidden_size]
        F_premises, F_hypotheses = util.split_after_concat(
            Fs3,
            self.batch_size)

        # [batch_size, timesteps, timesteps]
        eijs = tf.matmul(F_premises,
                         tf.transpose(F_hypotheses,
                                      perm=[0, 2, 1]),
                         name='eijs')

        # [batch_size, timesteps, timesteps]
        eijs_softmaxed = tf.nn.softmax(eijs)

        # [batch_size, timesteps, hidden_size]
        betas = tf.matmul(eijs_softmaxed,
                          self.hypotheses_encoding)

        # [batch_size, timesteps, hidden_size]
        alphas = tf.matmul(tf.transpose(eijs_softmaxed,
                                        perm=[0, 2, 1]),
                           self.premises_encoding)

        return betas, alphas

    @decorators.define_scope
    def compare(self):
        betas, alphas = self.align

        # [batch_size, timesteps, 2 * hidden_size]
        V1_input = tf.concat([self.premises_encoding,
                              betas],
                             axis=2)
        # [batch_size, timesteps, 2 * hidden_size]
        V2_input = tf.concat([self.hypotheses_encoding,
                              alphas],
                             axis=2)

        # [2 * batch_size, timesteps, 2 * hidden_size]
        ff_input = util.concat(V1_input, V2_input)
        Vs1 = model_base.fully_connected_with_dropout(
            inputs=ff_input,
            num_outputs=self.hidden_size,
            activation_fn=tf.nn.relu,
            p_keep=self.p_keep)
        Vs2 = model_base.fully_connected_with_dropout(
            inputs=Vs1,
            num_outputs=self.hidden_size,
            activation_fn=tf.nn.relu,
            p_keep=self.p_keep
        )
        Vs3 = model_base.fully_connected_with_dropout(
            inputs=Vs2,
            num_outputs=self.hidden_size,
            activation_fn=tf.nn.relu,
            p_keep=self.p_keep
        )

        # [batch_size, timesteps, hidden_size]
        V1, V2 = util.split_after_concat(Vs3, self.batch_size)

        return V1, V2


class BiRNNAlignment(Alignment):
    def __init__(self, config):
        self.p_keep_rnn = config['p_keep_rnn']
        Alignment.__init__(self, config)
        self.name = 'BiRNNAlign'

    @decorators.define_scope
    def premises_encoding(self):
        # [batch_size, timesteps, rnn_size]
        _premises_encoding = rnn_encoders.bi_rnn(
            sentences=self.premises,
            hidden_size=self.embed_size,
            scope='premise_bi_rnn',
            p_keep=self.p_keep_rnn)
        # [batch_size, timesteps, 2 * rnn_size]
        concatenated_encoding = tf.concat(_premises_encoding[0], 2)
        return concatenated_encoding

    @decorators.define_scope
    def hypotheses_encoding(self):
        # [batch_size, timesteps, rnn_size]
        _hypotheses_encoding = rnn_encoders.bi_rnn(
            sentences=self.hypotheses,
            hidden_size=self.embed_size,
            scope='hypothesis_bi_rnn',
            p_keep=self.p_keep_rnn)
        # [batch_size, timesteps, 2 * rnn_size]
        concatenated_encoding = tf.concat(_hypotheses_encoding[0], 2)
        return concatenated_encoding


class ChenAlignA(BiRNNAlignment):
    """
    Decisions in A:
    - no projection
    - not using LSTM in compare stage, ff instead
    """
    def __init__(self, config):
        BiRNNAlignment.__init__(self, config)
        self.name = 'ChenAlignA'
        self.weights_to_scale_factor = {
            self._weights(
                'premises_encoding/premise_bi_rnn/fw/basic_lstm_cell')
            : self.p_keep_rnn,
            self._weights(
                'premises_encoding/premise_bi_rnn/bw/basic_lstm_cell')
            : self.p_keep_rnn,
            self._weights(
                'hypotheses_encoding/hypothesis_bi_rnn/fw/basic_lstm_cell')
            : self.p_keep_rnn,
            self._weights(
                'hypotheses_encoding/hypothesis_bi_rnn/bw/basic_lstm_cell')
            : self.p_keep_rnn,
            self._weights(
                'compare/fully_connected')
            : self.p_keep,
            self._weights(
                'compare/fully_connected_1')
            : self.p_keep,
            self._weights(
                'logits/fully_connected')
            : self.p_keep,
            self._weights(
                'logits/fully_connected_1')
            : self.p_keep,
            self._weights(
                'logits/fully_connected_2')
            : self.p_keep
        }
        self.optimize_transfer

    @decorators.define_scope
    def align(self):
        # in the Chen model, we don't use a feedforward here
        # we are also skipping over projection

        # [batch_size, timesteps, timesteps]
        alignment_matrix = tf.matmul(
            self.premises_encoding,
            tf.transpose(self.hypotheses_encoding,
                         perm=[0, 2, 1]),
            name='alignment_matrix')

        # [batch_size, timesteps, hidden_size]
        premises_soft_aligment = tf.matmul(
            tf.nn.softmax(alignment_matrix),
            self.hypotheses_encoding)
        hypotheses_soft_alignment = tf.matmul(
            tf.nn.softmax(tf.transpose(alignment_matrix,
                                       perm=[0, 2, 1])),
            self.premises_encoding)

        # "Collection" phase

        # [batch_size, timesteps, 4 * hidden_size]
        premises_mimics = tf.concat(
            [self.premises_encoding,
             premises_soft_aligment,
             self.premises_encoding - premises_soft_aligment,
             tf.multiply(self.premises_encoding, premises_soft_aligment)],
            axis=2)
        # [batch_size, timesteps, 4 * hidden_size]
        hypotheses_mimics = tf.concat(
            [self.hypotheses_encoding,
             hypotheses_soft_alignment,
             self.hypotheses_encoding - hypotheses_soft_alignment,
             tf.multiply(self.hypotheses_encoding, hypotheses_soft_alignment)],
            axis=2)

        return premises_mimics, hypotheses_mimics

    @decorators.define_scope
    def compare(self):
        # [batch_size, timesteps, 4 * hidden_size]
        premises_mimics, hypotheses_mimics = self.align

        # [2 * batch_size, timesteps, 4 * hidden_size]
        ff_input = util.concat(premises_mimics, hypotheses_mimics)

        # [2 * batch_size, timesteps, hidden_size]
        h1 = model_base.fully_connected_with_dropout(
            inputs=ff_input,
            num_outputs=self.hidden_size,
            activation_fn=tf.nn.relu,
            p_keep=self.p_keep)
        h2 = model_base.fully_connected_with_dropout(
            inputs=h1,
            num_outputs=self.hidden_size,
            activation_fn=tf.nn.relu,
            p_keep=self.p_keep
        )

        # [batch_size, timesteps, hidden_size]
        v1, v2 = util.split_after_concat(h2, self.batch_size)

        return v1, v2

    @decorators.define_scope
    def optimize_transfer(self):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        weights_to_optimize = [w for w
                               in self._all_weights()
                               if w.name.startswith('logits')]
        grads_and_vars = optimizer.compute_gradients(
            self.loss,
            weights_to_optimize)
        if self.grad_clip_norm > 0.0:
            grads_and_vars = util.clip_gradients(grads_and_vars,
                                                 norm=self.grad_clip_norm)
        return optimizer.apply_gradients(grads_and_vars)


class ChenAlignB(ChenAlignA):
    """Alignment inspired by Chen (2017) version B.

    Decisions in B:
    - use projection
    - still no LSTM in compare stage, ff still

    https://arxiv.org/pdf/1609.06038v3.pdf
    """

    def __init__(self, config):
        ChenAlignA.__init__(self, config)
        self.name = 'ChenAlignB'

    @decorators.define_scope
    def align(self):
        """Compute alignment vectors and perform collection."""
        # in the Chen model, we don't use a feedforward here

        # this time do use projected word vectors
        projected_premises, projected_hypotheses = \
            util.split_after_concat(self.project, self.batch_size)

        # [batch_size, timesteps, timesteps]
        alignment_matrix = tf.matmul(
            projected_premises,
            tf.transpose(projected_hypotheses,
                         perm=[0, 2, 1]),
            name='alignment_matrix')

        # [batch_size, timesteps, hidden_size]
        premises_soft_aligment = tf.matmul(
            tf.nn.softmax(alignment_matrix),
            projected_hypotheses)
        hypotheses_soft_alignment = tf.matmul(
            tf.nn.softmax(tf.transpose(alignment_matrix,
                                       perm=[0, 2, 1])),
            projected_premises)

        # "Collection" phase

        # [batch_size, timesteps, 4 * hidden_size]
        premises_mimics = tf.concat(
            [projected_premises,
             premises_soft_aligment,
             projected_premises - premises_soft_aligment,
             tf.multiply(projected_premises, premises_soft_aligment)],
            axis=2)
        premises_mimics.set_shape([None,
                                   None,
                                   4 * self.hidden_size])
        # [batch_size, timesteps, 4 * hidden_size]
        hypotheses_mimics = tf.concat(
            [projected_hypotheses,
             hypotheses_soft_alignment,
             projected_hypotheses - hypotheses_soft_alignment,
             tf.multiply(projected_hypotheses, hypotheses_soft_alignment)],
            axis=2)
        hypotheses_mimics.set_shape([None,
                                     None,
                                     4 * self.hidden_size])

        return premises_mimics, hypotheses_mimics
