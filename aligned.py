import tensorflow as tf
import model_base
import decorators
import util
import rnn_encoders


# control randomization for reproducibility
tf.set_random_seed(1984)


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
        self.summary

    @decorators.define_scope
    def premises_encoding(self):
        # [batch_size, timesteps, embed_size]
        return self.X.premises

    @decorators.define_scope
    def hypotheses_encoding(self):
        # [batch_size, timesteps, embed_size]
        return self.X.hypotheses

    @decorators.define_scope
    def project(self):
        # [2 * batch_size, timesteps, embed_size]
        concatenated = util.concat(self.premises_encoding,
                                   self.hypotheses_encoding)

        # [2 * batch_size, timesteps, project_size]
        projected = model_base.fully_connected_with_dropout(
            concatenated,
            self.projection_size,
            self.activation,
            self.config.p_keep_ff)

        return projected

    @decorators.define_scope
    def align(self):
        # [2 * batch_size, timesteps, align_size]
        Fs1 = model_base.fully_connected_with_dropout(
             inputs=self.project,
             num_outputs=self.alignment_size,
             activation_fn=self.activation,
             p_keep=self.config.p_keep_ff)
        Fs2 = model_base.fully_connected_with_dropout(
            inputs=Fs1,
            num_outputs=self.alignment_size,
            activation_fn=self.activation,
            p_keep=self.config.p_keep_ff
        )

        # [batch_size, timesteps, align_size]
        F_premises, F_hypotheses = util.split_after_concat(
            Fs2,
            self.batch_size)

        # [batch_size, timesteps, timesteps]
        eijs = tf.matmul(F_premises,
                         tf.transpose(F_hypotheses,
                                      perm=[0, 2, 1]),
                         name='eijs')

        # [batch_size, timesteps, timesteps]
        eijs_softmaxed = tf.nn.softmax(eijs)

        # [batch_size, timesteps, align_size]
        betas = tf.matmul(eijs_softmaxed,
                          self.hypotheses_encoding)

        # [batch_size, timesteps, align_size]
        alphas = tf.matmul(tf.transpose(eijs_softmaxed,
                                        perm=[0, 2, 1]),
                           self.premises_encoding)

        return betas, alphas

    @decorators.define_scope
    def compare(self):
        betas, alphas = self.align
        concatenated = util.concat(betas, alphas)
        Vs1 = model_base.fully_connected_with_dropout(
             inputs=concatenated,
             num_outputs=self.config.ff_size,
             activation_fn=self.activation,
             p_keep=self.config.p_keep_ff)
        Vs2 = model_base.fully_connected_with_dropout(
            inputs=Vs1,
            num_outputs=self.config.ff_size,
            activation_fn=self.activation,
            p_keep=self.config.p_keep_ff
        )
        V1, V2 = util.split_after_concat(Vs2, self.batch_size)                         # [batch_size, timesteps, ff_size]
        return V1, V2

    @decorators.define_scope
    def aggregate(self):
        V1, V2 = self.compare                                                         # [batch_size, timesteps, ff_size]
        v1 = tf.reduce_sum(V1, axis=1)                                                        # [batch_size, 1, ff_size]
        v2 = tf.reduce_sum(V2, axis=1)                                                        # [batch_size, 1, ff_size]
        concatenated = tf.concat([v1, v2], axis=1)                         # [batch_size, 2, ff_size]  can't be right...
        concatenated.set_shape([None, 2 * self.config.ff_size])            # because this is 2d
        return concatenated                                                                  # [batch_size, 2 * ff_size]

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


class BiRNNAlignment(AlignmentParikh):
    def __init__(self, config, encoding_size=300, alignment_size=200, projection_size=200, activation=tf.nn.relu):
        AlignmentParikh.__init__(self, config, encoding_size, alignment_size, projection_size, activation)
        self.name = 'bi_rnn_alignment'

    @decorators.define_scope
    def premises_encoding(self):
        _premises_encoding = rnn_encoders.bi_rnn(self.X.premises,
                                                 self.config.word_embed_size,
                                                 'premise_bi_rnn',
                                                 self.config.p_keep_ff)              # [batch_size, timesteps, rnn_size]
        concatenated_encoding = tf.concat(_premises_encoding[0], 2)
        return concatenated_encoding                                             # [batch_size, timesteps, 2 * rnn_size]

    @decorators.define_scope
    def hypotheses_encoding(self):
        _hypotheses_encodings = rnn_encoders.bi_rnn(self.X.hypotheses,
                                                    self.config.word_embed_size,
                                                    'hypothesis_bi_rnn',
                                                    self.config.p_keep_ff)           # [batch_size, timesteps, rnn_size]
        concatenated_encoding = tf.concat(_hypotheses_encodings[0], 2)
        return concatenated_encoding                                             # [batch_size, timesteps, 2 * rnn_size]


if __name__ == '__main__':
    config = model_base.Config()
    model = AlignmentParikh(config)
    import numpy as np
    premises = np.random.rand(4, 12, 300)
    hypotheses = np.random.rand(4, 12, 300)
    labels = np.random.rand(4, 3)
    feed_dict = {model.X.premises: premises, model.X.hypotheses: hypotheses, model.Y: labels}
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(model.logits, feed_dict))
