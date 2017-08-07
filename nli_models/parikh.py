"""Parikh () alignment model."""
import tensorflow as tf
from coldnet.tensor_flow import models
from coldnet.tensor_flow import util as tf_util


class Parikh(models.TensorFlowModel):
    """Alignment model a la Parikh (2016).
    http://arxiv.org/pdf/1606.01933v1.pdf."""

    def __init__(self, config, vocab_dict):
        super(Parikh, self).__init__(config)
        self.vocab_dict = vocab_dict
        self.premise_ixs = \
            tf.placeholder(tf.int32, [None, None], 'premise_ixs')
        self.hypothesis_ixs = \
            tf.placeholder(tf.int32, [None, None], 'hypothesis_ixs')
        self.project
        self.align
        self.compare
        self.aggregate
        self._init_backend()  # logits called here

    @tf_util.define_scope
    def project(self):
        """Project the word vectors into a hidden_size-dimensional space.

        Concatenates the premises and hypotheses in the batch into
        a single tensor. This is required in the next step and is
        therefore convenient not to split it up upon returning here.

        Returns:
          Tensor of shape [2 * batch_size, timesteps, hidden_size].
        """
        # [2 * batch_size, timesteps, embed_size]
        concatenated = tf_util.concat(
            premises=self.premises,
            hypotheses=self.hypotheses)

        # [2 * batch_size, timesteps, hidden_size]
        projected = tf_util.fully_connected_with_dropout(
            inputs=concatenated,
            num_outputs=self.projection_size,
            activation_fn=None,
            keep_prob=self.p_keep_fc)

        return projected

    @tf_util.define_scope
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
        F1 = tf_util.fully_connected_with_dropout(
            inputs=self.project,
            num_outputs=self.hidden_size,
            activation_fn=tf.nn.relu,
            keep_prob=self.p_keep_fc)
        F2 = tf_util.fully_connected_with_dropout(
            inputs=F1,
            num_outputs=self.hidden_size,
            activation_fn=tf.nn.relu,
            keep_prob=self.p_keep_fc)

        # [batch_size, timesteps, hidden_size]
        F_premises, F_hypotheses = tf_util.split_after_concat(
            concatenated=F2,
            batch_size=self.current_batch_size)

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
                          self.hypotheses)

        # [batch_size, timesteps, hidden_size]
        alphas = tf.matmul(eijs_softmaxed_for_hypotheses,
                           self.premises)

        return betas, alphas

    @tf_util.define_scope
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
            [self.premises,
             betas],
            axis=2)
        # [batch_size, timesteps, 2 * hidden_size]
        hypotheses_input = tf.concat(
            [self.hypotheses,
             alphas],
            axis=2)

        # [2 * batch_size, timesteps, 2 * hidden_size]
        concatenated_input = tf_util.concat(
            premises=premises_input,
            hypotheses=hypotheses_input)
        dropped_input = tf.nn.dropout(
            x=concatenated_input,
            keep_prob=self.p_keep_input)

        G1 = tf_util.fully_connected_with_dropout(
            inputs=dropped_input,
            num_outputs=self.hidden_size,
            activation_fn=tf.nn.relu,
            keep_prob=self.p_keep_fc)
        G2 = tf_util.fully_connected_with_dropout(
            inputs=G1,
            num_outputs=self.hidden_size,
            activation_fn=tf.nn.relu,
            keep_prob=self.p_keep_fc)

        # [batch_size, timesteps, hidden_size]
        V1, V2 = tf_util.split_after_concat(G2, self.current_batch_size)

        return V1, V2

    @tf_util.define_scope
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
            [None, 2 * int(self.hidden_size)])

        dropped = tf.nn.dropout(
            x=concatenated,
            keep_prob=self.p_keep_input)

        return dropped

    @tf_util.define_scope
    def logits(self):
        """Define the vectors to pass to the classifer."""
        mlp1 = tf_util.fully_connected_with_dropout(
            inputs=self.aggregate,
            num_outputs=self.hidden_size,
            activation_fn=tf.nn.relu,
            keep_prob=self.p_keep_fc)
        mlp2 = tf_util.fully_connected_with_dropout(
            inputs=mlp1,
            num_outputs=self.hidden_size,
            activation_fn=tf.nn.relu,
            keep_prob=self.p_keep_fc)
        logits = tf_util.fully_connected_with_dropout(
            inputs=mlp2,
            num_outputs=3,
            activation_fn=None,
            keep_prob=1.0)
        return logits
