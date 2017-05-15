import tensorflow as tf
import decorators
import model_base
import util


def bi_rnn(sentences, hidden_size, scope,
           p_keep=0.8, cell=tf.contrib.rnn.BasicLSTMCell):
    sequence_length = util.length(sentences)
    forward_cell = cell(
        hidden_size,
        forget_bias=1.0)
    forward_cell = tf.contrib.rnn.DropoutWrapper(
        cell=forward_cell,
        output_keep_prob=p_keep)
    backward_cell = cell(
        hidden_size,
        forget_bias=1.0)
    backward_cell = tf.contrib.rnn.DropoutWrapper(
        cell=backward_cell,
        output_keep_prob=p_keep)
    output, output_states = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=forward_cell,
        cell_bw=backward_cell,
        inputs=sentences,
        sequence_length=sequence_length,
        dtype=tf.float64,
        scope=scope)
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
        self.logits
        self.loss
        self._init_backend()

    @decorators.define_scope
    def premises_encoding(self):
        return self.premises

    @decorators.define_scope
    def hypotheses_encoding(self):
        return self.hypotheses

    @decorators.define_scope
    def logits(self):
        concatenated_encodings = tf.concat(
            [self.premises_encoding, self.hypotheses_encoding],
            axis=1,
            name='concatenated_encodings')
        a1 = model_base.fully_connected_with_dropout(
            inputs=concatenated_encodings,
            num_outputs=self.hidden_size,
            activation_fn=tf.tanh,
            p_keep=self.p_keep
        )
        a2 = model_base.fully_connected_with_dropout(
            inputs=a1,
            num_outputs=self.hidden_size,
            activation_fn=tf.tanh,
            p_keep=self.p_keep
        )
        a3 = model_base.fully_connected_with_dropout(
            inputs=a2,
            num_outputs=self.hidden_size,
            activation_fn=tf.tanh,
            p_keep=self.p_keep
        )
        logits = tf.contrib.layers.fully_connected(
            inputs=a3,
            num_outputs=3,
            activation_fn=None)
        return logits


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
        _premises_encoding = bi_rnn(
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
        _hypotheses_encoding = bi_rnn(
            self.hypotheses,
            self.embed_size,
            'hypothesis_bi_rnn',
            self.p_keep)
        # [batch_size, timesteps, 2 * rnn_size]
        concatenated_encoding = tf.concat(_hypotheses_encoding[0], 2)
        return concatenated_encoding


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
        labels.set_shape([self.batch_size, 1])
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
