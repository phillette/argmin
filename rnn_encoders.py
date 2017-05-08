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
