import tensorflow as tf

import decorators
import model_base
from argmin import util

# control randomization for reproducibility
tf.set_random_seed(1984)


class BinaryTreeLSTMCell(tf.contrib.rnn.BasicLSTMCell):
    """LSTM with two state inputs.

    Code from tensorflow fold tutorial:
    https://github.com/tensorflow/fold/blob/
    master/tensorflow_fold/g3doc/sentiment.ipynb

    This is the model described in section 3.2 of 'Improved Semantic
    Representations From Tree-Structured Long Short-Term Memory
    Networks' <http://arxiv.org/pdf/1503.00075.pdf>, with recurrent
    dropout as described in 'Recurrent Dropout without Memory Loss'
    <http://arxiv.org/pdf/1603.05118.pdf>.
    """

    def __init__(self, num_units, keep_prob=1.0):  # make this a ph?
        """Initialize the cell.

        Args:
          num_units: int, The number of units in the LSTM cell.
          keep_prob: Keep probability for recurrent dropout.
        """
        # super allows non-explicit reference to the base class
        super(BinaryTreeLSTMCell, self).__init__(num_units)
        self._keep_prob = keep_prob

    def __call__(self, inputs, state, scope=None):
        # this "or" is better than a ternary operator!
        with tf.variable_scope(scope or type(self).__name__):
            lhs, rhs = state
            c0, h0 = lhs
            c1, h1 = rhs
            # linear seems to be deprecated - tf.contrib.layers.fully_connected
            # with activation_fn=None will achieve the same effect
            concat = tf.contrib.layers.linear(
                tf.concat([inputs, h0, h1], 1), 5 * self._num_units)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f0, f1, o = tf.split(value=concat, num_or_size_splits=5, axis=1)

            j = self._activation(j)
            if not isinstance(self._keep_prob, float) or self._keep_prob < 1:
                j = tf.nn.dropout(j, self._keep_prob)

            new_c = (c0 * tf.sigmoid(f0 + self._forget_bias) +
                c1 * tf.sigmoid(f1 + self._forget_bias) +
                tf.sigmoid(i) * j)
            new_h = self._activation(new_c) * tf.sigmoid(o)

            new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

            return new_h, new_state


def bi_rnn(sentences, hidden_size, scope,
           dropout_config, dropout_key='rnn',
           cell=tf.contrib.rnn.BasicLSTMCell):
    sequence_length = util.length(sentences)
    forward_cell = cell(
        int(hidden_size / dropout_config.raw[dropout_key]),
        forget_bias=1.0)
    forward_cell = tf.contrib.rnn.DropoutWrapper(
        cell=forward_cell,
        output_keep_prob=dropout_config.ops[dropout_key])
    backward_cell = cell(
        int(hidden_size / dropout_config.raw[dropout_key]),
        forget_bias=1.0)
    backward_cell = tf.contrib.rnn.DropoutWrapper(
        cell=backward_cell,
        output_keep_prob=dropout_config.ops[dropout_key])
    output, output_states = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=forward_cell,
        cell_bw=backward_cell,
        inputs=sentences,
        sequence_length=sequence_length,
        dtype=tf.float64,
        scope=scope)
    # output is [batch_size, timesteps, hidden_size]
    # in a tuple, seems [0] is forward and [1] is backward
    # output_states: LSTMTupleSTate, again a tuple of them,
    # [0] probably forward [1] probably back.
    # the state itself has two vectors c and h. They are identical
    # being of size [batch_size, hidden_size]. One of them is the
    # final state vector, but I don't rightly know which...
    return output, output_states


def lstm_encoder(sentences, hidden_size, scope, p_keep=0.8):
    sequence_length = util.length(sentences)
    cell = tf.contrib.rnn.BasicLSTMCell(
        hidden_size,
        forget_bias=1.0)
    cell = tf.contrib.rnn.DropoutWrapper(
        cell=cell,
        output_keep_prob=p_keep)
    output, output_states = tf.nn.dynamic_rnn(
        cell=cell,
        inputs=sentences,
        sequence_length=sequence_length,
        dtype=tf.float64,
        scope=scope)
    return output, output_states


class Encoder(model_base.Model):
    def __init__(self, config):
        model_base.Model.__init__(self, config)
        self.name = 'encoder'
        self.premises_encoding
        self.hypotheses_encoding
        self._init_backend()

    @decorators.define_scope
    def premises_encoding(self):
        raise NotImplementedError()

    @decorators.define_scope
    def hypotheses_encoding(self):
        raise NotImplementedError()

    @decorators.define_scope
    def classifier_input(self):
        input = tf.concat(
            [self.premises_encoding,
             self.hypotheses_encoding,
             tf.abs(tf.subtract(
                 self.premises_encoding,
                 self.hypotheses_encoding)),
             tf.multiply(
                 self.premises_encoding,
                 self.hypotheses_encoding)],
            axis=1,
            name='concatenated_encodings')
        dropped_input = tf.nn.dropout(
            x=input,
            keep_prob=self.dropout_config.ops['input'])
        return dropped_input

    @decorators.define_scope
    def logits(self):
        # this currently overloads model_base.Model.logits;
        # it is probably redundant now.
        h1 = tf.contrib.layers.fully_connected(
            inputs=self.classifier_input,
            # looking at this now I think this should be in another config...
            num_outputs=int(self.hidden_size / self.dropout_config.raw['ff']),
            activation_fn=tf.tanh)
        self.h1_dropped = tf.nn.dropout(
            x=h1,
            keep_prob=self.dropout_config.ops['ff'])
        h2 = tf.contrib.layers.fully_connected(
            inputs=self.h1_dropped,
            num_outputs=int(self.hidden_size / self.dropout_config.raw['ff']),
            activation_fn=tf.tanh)
        h2_dropped = tf.nn.dropout(
            x=h2,
            keep_prob=self.dropout_config.ops['ff'])
        _logits = tf.contrib.layers.fully_connected(
            inputs=h2_dropped,
            num_outputs=3,
            activation_fn=None)
        return _logits


class LSTMEncoder(Encoder):
    def __init__(self, config):
        Encoder.__init__(self, config)
        self.name = 'lstm_encoder'

    @decorators.define_scope
    def premises_encoding(self):
        encoding = lstm_encoder(
            sentences=self.premises,
            hidden_size=self.hidden_size,
            scope='premises_encoder',
            p_keep=self.p_keep
        )
        concatenated_encoding = tf.concat(encoding[0], 2)
        return concatenated_encoding

    @decorators.define_scope
    def hypotheses_encoding(self):
        encoding = lstm_encoder(
            sentences=self.hypotheses,
            hidden_size=self.hidden_size,
            scope='hypotheses_encoder',
            p_keep=self.p_keep
        )
        concatenated_encoding = tf.concat(encoding[0], 2)
        return concatenated_encoding


class BiLSTMEncoder(Encoder):
    def __init__(self, config):
        Encoder.__init__(self, config)
        self.name = 'BiLSTMEnc'

    @decorators.define_scope
    def premises_encoding(self):
        # [batch_size, timesteps, rnn_size]
        output, _ = bi_rnn(
            sentences=self.premises,
            hidden_size=self.hidden_size,
            scope='premise_bi_rnn',
            dropout_config=self.dropout_config)
        concatenated = tf.concat(output, axis=2)
        max_pooled = tf.reduce_max(concatenated, axis=1)
        return max_pooled

    @decorators.define_scope
    def hypotheses_encoding(self):
        # [batch_size, timesteps, rnn_size]
        output, _ = bi_rnn(
            sentences=self.hypotheses,
            hidden_size=self.hidden_size,
            scope='hypothesis_bi_rnn',
            dropout_config=self.dropout_config)
        concatenated = tf.concat(output, axis=2)
        max_pooled = tf.reduce_max(concatenated, axis=1)
        return max_pooled

    @decorators.define_scope
    def linear_logits(self):
        return tf.contrib.layers.fully_connected(
            inputs=self.h1_dropped,
            num_outputs=self.config['linear_logits_output'],
            activation_fn=None)


class SimpleEncoder(model_base.Model):
    """My attempt at a simple encoding model."""

    def __init__(self, config):
        model_base.Model.__init__(self, config)
        self.name = 'SimpleEncoder'
        self.premises_encoding
        self.hypotheses_encoding
        self.premises_projection
        self.hypotheses_projection
        self.attended_premises
        self.attended_hypotheses
        self.final_premises_encoding
        self.final_hypotheses_encoding
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
        """Encode the premises via BiRNN."""
        # [batch_size, timesteps, embed_size]
        encodings = bi_rnn(
            self.premises,
            self.embed_size,
            'premise_bi_rnn',
            self.p_keep)
        # [batch_size, timesteps, 2 * embed_size]
        concatenated_encoding = tf.concat(encodings[0], 2)
        return concatenated_encoding

    @decorators.define_scope
    def hypotheses_encoding(self):
        """Encode the hypotheses via BiRNN."""
        # [batch_size, timesteps, embed_size]
        encodings = bi_rnn(
            self.hypotheses,
            self.embed_size,
            'hypothesis_bi_rnn',
            self.p_keep)
        # [batch_size, timesteps, 2 * embed_size]
        concatenated_encoding = tf.concat(encodings[0], 2)
        return concatenated_encoding

    @decorators.define_scope
    def project(self):
        ff_input = tf.concat([self.premises_encoding,
                              self.hypotheses_encoding],
                             axis=0)
        projection = model_base.fully_connected_with_dropout(
            inputs=ff_input,
            num_outputs=self.hidden_size,
            activation_fn=None,  # projection layer should not have activation!
            p_keep=1.0)  # no need to drop here

        return projection

    @decorators.define_scope('project')
    def premises_projection(self):
        """Project the premises encoding."""
        projected_premises, _ = util.split_after_concat(self.project,
                                                        self.batch_size)
        return projected_premises

    @decorators.define_scope('project')
    def hypotheses_projection(self):
        """Project the hypotheses encoding."""
        _, projected_hypotheses = util.split_after_concat(self.project,
                                                          self.batch_size)
        return projected_hypotheses

    @decorators.define_scope
    def attended_premises(self):
        """Get attention weighted vectors for premises."""
        # [batch_size, timesteps, timesteps]
        attention_weights_premises = tf.matmul(
            self.premises_projection,
            tf.transpose(self.premises_projection,
                         perm=[0, 2, 1]),
            name='attention_weights_premises')

        # []
        soft_attention_premises = tf.nn.softmax(attention_weights_premises)

        # []
        attended = tf.matmul(
            soft_attention_premises,
            self.premises_projection)
        attended.set_shape([None,
                            None,
                            self.hidden_size])

        return attended

    @decorators.define_scope
    def attended_hypotheses(self):
        """Get attention weighted vectors for hypotheses."""
        # [batch_size, timesteps, timesteps]
        attention_weights_hypotheses = tf.matmul(
            self.hypotheses_projection,
            tf.transpose(self.hypotheses_projection,
                         perm=[0, 2, 1]),
            name='attention_weights_hypotheses')

        soft_attention_hypotheses = tf.nn.softmax(attention_weights_hypotheses)

        attended = tf.matmul(
            soft_attention_hypotheses,
            self.hypotheses_projection)
        attended.set_shape([None,
                            None,
                            self.hidden_size])

        return attended

    @decorators.define_scope
    def final_premises_encoding(self):
        """Encode the premises from attended vectors via BiRNN."""
        # [batch_size, timesteps, hidden_size]
        encodings = bi_rnn(
            self.attended_premises,
            self.hidden_size,
            'final_premises_encoding_bi_rnn',
            self.p_keep)
        # [batch_size, timesteps, 2 * embed_size]
        concatenated_encoding = tf.concat(encodings[0], 2)
        return concatenated_encoding

    @decorators.define_scope
    def final_hypotheses_encoding(self):
        """Encode the hypotheses from attended vectors via BiRNN."""
        # [batch_size, timesteps, hidden_size]
        encodings = bi_rnn(
            self.attended_hypotheses,
            self.hidden_size,
            'final_hypotheses_encoding_bi_rnn',
            self.p_keep)
        # [batch_size, timesteps, 2 * embed_size]
        concatenated_encoding = tf.concat(encodings[0], 2)
        return concatenated_encoding

    @decorators.define_scope
    def logits(self):
        # what about attending over the vectors here?
        ff_input = tf.concat([self.final_premises_encoding,
                              self.final_hypotheses_encoding],
                             axis=2)
        h1 = model_base.fully_connected_with_dropout(
            inputs=ff_input,
            num_outputs=self.hidden_size,
            activation_fn=tf.tanh,
            p_keep=self.p_keep)
        h2 = model_base.fully_connected_with_dropout(
            inputs=h1,
            num_outputs=self.hidden_size,
            activation_fn=tf.tanh,
            p_keep=self.p_keep)
        return tf.contrib.layers.fully_connected(h2, 3, None)

    @decorators.define_scope
    def loss(self):
        labels = tf.argmax(self.Y, axis=1)
        labels = tf.reshape(labels, [32, 1])
        cross_entropy = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels,
                logits=self.logits,
                name='softmax_cross_entropy'))
        penalty_term = tf.multiply(
            tf.cast(self.lamda, tf.float64),
            sum([tf.nn.l2_loss(w) for w in self._all_weights()]),
            name='penalty_term')
        return tf.add(cross_entropy, penalty_term, name='loss')
