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
        self.name = 'alignment2'
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
        concatenated = util.concat(self.premises_encoding,
                                   self.hypotheses_encoding)

        # [2 * batch_size, timesteps, hidden_size]
        projected = model_base.fully_connected_with_dropout(
            concatenated,
            self.hidden_size,
            tf.nn.relu,
            self.p_keep)

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
            Fs2,
            self.batch_size)

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
        a1 = model_base.fully_connected_with_dropout(self.aggregate,
                                                     self.hidden_size,
                                                     tf.nn.relu,
                                                     self.p_keep)
        a2 = model_base.fully_connected_with_dropout(a1,
                                                     self.hidden_size,
                                                     tf.nn.relu,
                                                     self.p_keep)
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


class BiRNNAlignment(AlignmentDeep):
    def __init__(self, config):
        Alignment.__init__(self, config)
        self.name = 'bi_rnn_alignment'

    @decorators.define_scope
    def premises_encoding(self):
        # [batch_size, timesteps, rnn_size]
        _premises_encoding = rnn_encoders.bi_rnn(
            self.premises,
            self.embed_size,
            'premise_bi_rnn',
            self.p_keep)
        # [batch_size, timesteps, 2 * rnn_size]
        concatenated_encoding = tf.concat(_premises_encoding[0], 2)
        return concatenated_encoding

    @decorators.define_scope
    def hypotheses_encoding(self):
        # [batch_size, timesteps, rnn_size]
        _hypotheses_encoding = rnn_encoders.bi_rnn(
            self.hypotheses,
            self.embed_size,
            'hypothesis_bi_rnn',
            self.p_keep)
        # [batch_size, timesteps, 2 * rnn_size]
        concatenated_encoding = tf.concat(_hypotheses_encoding[0], 2)
        return concatenated_encoding


if __name__ == '__main__':
    config = model_base.base_config()
    model = Alignment(config)
    import numpy as np
    premises = np.random.rand(4, 12, 300)
    hypotheses = np.random.rand(4, 12, 300)
    labels = np.random.rand(4, 3)
    feed_dict = {model.premises: premises,
                 model.hypotheses: hypotheses,
                 model.Y: labels}
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        weights = model._all_weights()
        for weight in weights:
            print('%s: %s' % (weight.name, sess.run(tf.shape(weight))))
