"""Models that use Alignment between sentences."""
import tensorflow as tf

import decorators
import model_base
import rnn_encoders
from argmin import util

# control randomization for reproducibility
tf.set_random_seed(1984)


class Alignment(model_base.Model):
    """Alignment model a la Parikh (2016).

    http://arxiv.org/pdf/1606.01933v1.pdf.
    """

    def __init__(self, config):
        """Create a new Alignment model.

        Args:
          config: model_base.config dictionary.
        """
        config['activation_fn'] = 'relu'
        model_base.Model.__init__(self, config)
        self.name = 'alignment'
        self.premises_encoding
        self.hypotheses_encoding
        self.project
        self.align
        self.compare
        self.aggregate
        self._init_backend()

    @decorators.define_scope
    def premises_encoding(self):
        """The final encoding of premises.

        This op is introduced as an extensible part of
        the model - e.g. adding an BiLSTM to the front.

        Returns:
          Tensor of dimension [batch_size, timesteps, embed_size]
        """
        return self.premises

    @decorators.define_scope
    def hypotheses_encoding(self):
        """The final encoding of the hypotheses.

        This op is introduced as an extensible part of
        the model - e.g. adding an BiLSTM to the front.

        Returns:
          Tensor of dimension [batch_size, timesteps, embed_size]
        """
        return self.hypotheses

    @decorators.define_scope
    def project(self):
        """Project the word vectors into a hidden_size-dimensional space.

        Concatenates the premises and hypotheses in the batch into
        a single tensor. This is required in the next step and is
        therefore convenient not to split it up upon returning here.

        Returns:
          Tensor of shape [2 * batch_size, timesteps, hidden_size].
        """
        # [2 * batch_size, timesteps, embed_size]
        concatenated = util.concat(
            premises=self.premises_encoding,
            hypotheses=self.hypotheses_encoding)

        # [2 * batch_size, timesteps, hidden_size]
        projected = model_base.fully_connected_with_dropout(
            inputs=concatenated,
            num_outputs=self.projection_size,
            activation_fn=None,
            dropout_config=self.dropout_config,
            dropout_key='input',  # decided to consider this as input, not ff.
            scale_output_size=False)  # don't scale up the projection matrix.

        return projected

    @decorators.define_scope
    def align(self):
        """Align relevant parts of the two sentences.

        First pass the projected vectors through a function, F,
        (a feedforward neural network).

        Then perform a matrix multiplication of the result vectors
        (premises vector matrix dot the transpose of the hypotheses)
        to perform similarity matching, yielding raw relevance
        scores (eijs).

        Then apply softmax to smooth these relevance scores,
        remembering the orientation needs to be transposed for
        the hypotheses.

        The relevance scores are then applied to the vectors from
        the opposite sentence to scale them.

        QUESTIONS:
        - How many layers do we really need at the start of
          this part of the model? Parikh's paper reports two.

        Returns:
          alphas, betas: both tensors of shape
            [batch_size, timesteps, hidden_size].
        """
        # [2 * batch_size, timesteps, hidden_size]
        F1 = model_base.fully_connected_with_dropout(
            inputs=self.project,
            num_outputs=self.hidden_size,
            activation_fn=tf.nn.relu,
            dropout_config=self.dropout_config,
            dropout_key='ff')
        F2 = model_base.fully_connected_with_dropout(
            inputs=F1,
            num_outputs=self.hidden_size,
            activation_fn=tf.nn.relu,
            dropout_config=self.dropout_config,
            dropout_key='ff')

        # [batch_size, timesteps, hidden_size]
        F_premises, F_hypotheses = util.split_after_concat(
            tensor=F2,
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
        """Compute comparison vectors between the sentences.

        The comparison vectors are non-linear combinations
        of the original sentences and their aligned opposite
        vectors (the alphas and betas from the align step).

        We start by concatenating the original vectors with
        the alphas and betas. We then pass these through a
        function, G, again a (two layer) feedforward network.

        Returns:
          V1, V2: comparison vectors, both tensors of shape
            [batch_size, timesteps, hidden_size].
        """
        betas, alphas = self.align

        # [batch_size, timesteps, 2 * hidden_size]
        premises_input = tf.concat(
            [self.premises_encoding,
             betas],
            axis=2)
        # [batch_size, timesteps, 2 * hidden_size]
        hypotheses_input = tf.concat(
            [self.hypotheses_encoding,
             alphas],
            axis=2)

        # [2 * batch_size, timesteps, 2 * hidden_size]
        concatenated_input = util.concat(
            premises=premises_input,
            hypotheses=hypotheses_input)
        dropped_input = tf.nn.dropout(
            x=concatenated_input,
            keep_prob=self.dropout_config.ops['input'])

        G1 = model_base.fully_connected_with_dropout(
            inputs=dropped_input,
            num_outputs=self.hidden_size,
            activation_fn=tf.nn.relu,
            dropout_config=self.dropout_config,
            dropout_key='ff')
        G2 = model_base.fully_connected_with_dropout(
            inputs=G1,
            num_outputs=self.hidden_size,
            activation_fn=tf.nn.relu,
            dropout_config=self.dropout_config,
            dropout_key='ff')

        # [batch_size, timesteps, hidden_size]
        V1, V2 = util.split_after_concat(G2, self.batch_size)

        return V1, V2

    @decorators.define_scope
    def aggregate(self):
        """Aggregate the comparison vectors.

        This step combines the comparison vectors (V1 and V2)
        and prepares them to be passed to the classifier.

        In Parikh's original paper aggregation is performed as
        a sum of each vector.

        A potentially better aggregation method was introduced by
        Chen 2016 (http://arxiv.org/pdf/1609.06038v3.pdf):
        to concatenate the averages and maxs of the two sets
        of comparison vectors.

        For the transfer learning experiments I have gone with
        the simpler summation because it involves half the
        parameters in the first feedforward layer. We are
        transferring to small data sets, and the point of
        putting this model back into comparison was its lean
        and effective structure. In any case, 86% on SNLI is
        a decent result for this model with summation.

        Returns:
          Tensor of shape [batch_size, 4 * self.hidden_size]
        """
        # [batch_size, timesteps, hidden_size]
        V1, V2 = self.compare

        sum_premises = tf.reduce_sum(V1, axis=1)
        sum_hypotheses = tf.reduce_sum(V2, axis=1)
        concatenated = tf.concat([sum_premises,
                                  sum_hypotheses],
                                 axis=1)
        concatenated.set_shape(
            [None, 2 * int(self.hidden_size / self.config['p_keep_ff'])])

        # new aggregation method (Chen)
        #avg_premises = tf.reduce_mean(V1, axis=1)
        #max_premises = tf.reduce_max(V1, axis=1)
        #avg_hypotheses = tf.reduce_mean(V2, axis=1)
        #max_hypotheses = tf.reduce_max(V2, axis=1)
        #concatenated = tf.concat([avg_premises,
        #                          max_premises,
        #                          avg_hypotheses,
        #                          max_hypotheses],
        #                         axis=1)
        # [batch_size, 4 * hidden_size] (that's 2 * hidden_size per sentence)
        #concatenated.set_shape(
        #    [None, 4 * int(self.hidden_size / self.config['p_keep_ff'])])

        dropped = tf.nn.dropout(
            x=concatenated,
            keep_prob=self.dropout_config.ops['input'])

        return dropped

    @decorators.define_scope
    def classifier_input(self):
        """Define the vectors to pass to the classifer."""
        return self.aggregate

    #@decorators.define_scope
    #def optimize(self):
    #    optimizer = \
    #        tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)
    #    return optimizer


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
            dropout_config=self.dropout_config,
            dropout_key='rnn')
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
            dropout_config=self.dropout_config,
            dropout_key='rnn')
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
            dropout_config=self.dropout_config,
            dropout_key='ff')
        h2 = model_base.fully_connected_with_dropout(
            inputs=h1,
            num_outputs=self.hidden_size,
            activation_fn=tf.nn.relu,
            dropout_config=self.dropout_config,
            dropout_key='ff')

        # [batch_size, timesteps, hidden_size]
        v1, v2 = util.split_after_concat(h2, self.batch_size)

        return v1, v2

    @decorators.define_scope
    def aggregate(self):
        # [batch_size, timesteps, hidden_size]
        V1, V2 = self.compare

        avg_premises = tf.reduce_mean(V1, axis=1)
        max_premises = tf.reduce_max(V1, axis=1)
        avg_hypotheses = tf.reduce_mean(V2, axis=1)
        max_hypotheses = tf.reduce_max(V2, axis=1)
        concatenated = tf.concat([avg_premises,
                                  max_premises,
                                  avg_hypotheses,
                                  max_hypotheses],
                                 axis=1)
        # [batch_size, 4 * hidden_size] (that's 2 * hidden_size per sentence)
        concatenated.set_shape(
           [None, 4 * int(self.hidden_size / self.config['p_keep_ff'])])

        dropped = tf.nn.dropout(
            x=concatenated,
            keep_prob=self.dropout_config.ops['input'])

        return dropped
