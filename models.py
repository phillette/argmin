import tensorflow as tf
from tensorflow.contrib import rnn
from tf_decorators import define_scope
from process_data import LONGEST_SENTENCE_SNLI, NUM_LABELS, get_batch_gen
from util import clip_gradients, length, feed_dict


def bi_rnn(sentences, hidden_size, scope, p_keep=0.8):
    sequence_length = length(sentences)
    forward_cell = rnn.BasicLSTMCell(hidden_size, forget_bias=1.0)
    forward_cell = tf.contrib.rnn.DropoutWrapper(cell=forward_cell,
                                                 output_keep_prob=p_keep)
    backward_cell = rnn.BasicLSTMCell(hidden_size, forget_bias=1.0)
    backward_cell = tf.contrib.rnn.DropoutWrapper(cell=backward_cell,
                                                  output_keep_prob=p_keep)
    output, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=forward_cell,
                                                            cell_bw=backward_cell,
                                                            inputs=sentences,
                                                            sequence_length=sequence_length,
                                                            dtype=tf.float64,
                                                            scope=scope)
    return output, output_states


def lstm_encoder(sentences, hidden_size, scope, p_dropout=0.8):
    sequence_length = length(sentences)
    cell = rnn.BasicLSTMCell(hidden_size, forget_bias=1.0)
    cell = tf.contrib.rnn.DropoutWrapper(cell=cell,
                                         output_keep_prob=p_dropout)
    output, output_states = tf.nn.dynamic_rnn(cell=cell,
                                              inputs=sentences,
                                              sequence_length=sequence_length,
                                              dtype=tf.float64,
                                              scope=scope)
    return output, output_states


class Model:
    def __init__(self, word_embed_length=300, learning_rate=0.001, hidden_size=100,
                 p_keep_input=0.8, p_keep_hidden=0.5, grad_norm=3.0):
        self.word_embed_length = word_embed_length
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.time_steps = LONGEST_SENTENCE_SNLI
        self.p_keep_input = p_keep_input
        self.p_keep_hidden = p_keep_hidden
        self.grad_norm = grad_norm
        self.global_step = tf.Variable(0,
                                       dtype=tf.int32,
                                       trainable=False,
                                       name='global_step')
        self._data
        self.logits
        self.loss
        self.optimize
        self.accuracy_train
        self.accuracy

    @define_scope()
    def accuracy_train(self):
        return self.accuracy

    @define_scope()
    def accuracy(self):
        correct_predictions = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float64))
        return accuracy

    @define_scope('data')
    def _data(self):
        self.premises = tf.placeholder(tf.float64,
                                       [None,
                                        self.time_steps,
                                        self.word_embed_length],
                                       name='premises')
        self.hypotheses = tf.placeholder(tf.float64,
                                         [None,
                                          self.time_steps,
                                          self.word_embed_length],
                                         name='hypotheses')
        self.y = tf.placeholder(tf.float64,
                                [None, 3],
                                name='y')
        return self.premises, self.hypotheses, self.y

    @define_scope
    def loss(self):
        return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.y,
                                                                     logits=self.logits,
                                                                     name='loss'))

    @define_scope
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        weights = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('weights:0')]
        grads_and_vars = optimizer.compute_gradients(self.loss, weights)
        clipped_grads_and_vars = clip_gradients(grads_and_vars)
        return optimizer.apply_gradients(clipped_grads_and_vars)


class LSTMEncoder(Model):
    def __init__(self, word_embed_length=300, learning_rate=0.001, hidden_size=100,
                 p_keep_input=0.8, p_keep_hidden=0.5, grad_norm=3.0):
        Model.__init__(self, word_embed_length, learning_rate, hidden_size,
                       p_keep_input, p_keep_hidden, grad_norm)
        self.name = 'lstm_encoder'

    @define_scope
    def logits(self):
        _, self.premise_encoding = lstm_encoder(self.premises,       # batch_size x hidden_size
                                                self.hidden_size,
                                                'premise_encoding',
                                                self.p_keep_input)
        _, self.hypothesis_encoding = lstm_encoder(self.hypotheses,  # batch_size x hidden_size
                                                   self.hidden_size,
                                                   'hypothesis_encoding',
                                                   self.p_keep_input)
        self.premise_reduced_encoding = tf.contrib.layers.fully_connected(inputs=self.premise_encoding.c,
                                                                          num_outputs=100,
                                                                          activation_fn=tf.tanh)
        self.hypothesis_reduced_encoding = tf.contrib.layers.fully_connected(inputs=self.hypothesis_encoding.c,
                                                                             num_outputs=100,
                                                                             activation_fn=tf.tanh)
        self.concatenated_encodings = tf.concat([self.premise_reduced_encoding, self.hypothesis_reduced_encoding],
                                                axis=1,
                                                name='concatenated_encodings')
        self.tanh1 = tf.contrib.layers.fully_connected(inputs=self.concatenated_encodings,
                                                       num_outputs=2 * self.hidden_size,
                                                       activation_fn=tf.tanh)
        self.tanh2 = tf.contrib.layers.fully_connected(inputs=self.tanh1,
                                                       num_outputs=2 * self.hidden_size,
                                                       activation_fn=tf.tanh)
        self.tanh3 = tf.contrib.layers.fully_connected(inputs=self.tanh2,
                                                       num_outputs=NUM_LABELS,
                                                       activation_fn=tf.tanh)
        return self.tanh3


class BiRNN:
    """
    Learning rate: 1e-3
    Training set accuracy = 0.969
    Test set accuracy = 0.6828
    Carstens = 0.46 (not fully trained but best average accuracy of a batch before gradients explode)
    """
    def __init__(self, word_embed_length=300, learning_rate=0.001, rnn_size=100, ff_size=200,
                 p_keep_rnn=0.8, p_keep_ff=0.5, grad_clip_norm=5.0):
        self.name = 'bi_rnn'
        self.word_embed_length = word_embed_length
        self.learning_rate = learning_rate
        self.rnn_size = rnn_size
        self.ff_size = ff_size
        self.p_keep_rnn = p_keep_rnn
        self.p_keep_ff = p_keep_ff
        self.grad_clip_norm = grad_clip_norm
        self.time_steps = LONGEST_SENTENCE_SNLI
        self.global_step = tf.Variable(0,
                                       dtype=tf.int32,
                                       trainable=False,
                                       name='global_step')
        self._data
        self._bi_rnns
        self._logits
        self.loss
        self.optimize
        self.accuracy_train
        self.accuracy

    @define_scope()
    def accuracy_train(self):
        return self.accuracy

    @define_scope()
    def accuracy(self):
        correct_predictions = tf.equal(tf.argmax(self._logits, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float64))
        return accuracy

    @define_scope('bi_rnns')
    def _bi_rnns(self):
        _, self.premise_output_states = bi_rnn(self.premises,
                                               self.rnn_size,
                                               'premise_bi_rnn',
                                               self.p_keep_rnn)
        self.premise_out = tf.concat([state.c for state in self.premise_output_states], axis=1)
        _, self.hypothesis_output_states = bi_rnn(self.hypotheses,
                                                  self.rnn_size,
                                                  'hypothesis_bi_rnn',
                                                  self.p_keep_rnn)
        self.hypothesis_out = tf.concat([state.c for state in self.hypothesis_output_states], axis=1)
        self.rnn_output = tf.concat([self.premise_out, self.hypothesis_out],
                                    axis=1,
                                    name='concatenated_sentences')
        return self.rnn_output  # batch_size x (4 * hidden_size)

    @define_scope('data')
    def _data(self):
        self.premises = tf.placeholder(tf.float64,
                                       [None,
                                        self.time_steps,
                                        self.word_embed_length],
                                       name='premises')
        self.hypotheses = tf.placeholder(tf.float64,
                                         [None,
                                          self.time_steps,
                                          self.word_embed_length],
                                         name='hypotheses')
        self.y = tf.placeholder(tf.float64,
                                [None, 3],
                                name='y')
        return self.premises, self.hypotheses, self.y

    @define_scope('feedforward')
    def _logits(self):
        self.hidden_output_1 = tf.contrib.layers.fully_connected(self.rnn_output,
                                                                 self.ff_size,
                                                                 tf.tanh)
        self.hidden_1_dropped = tf.nn.dropout(self.hidden_output_1,
                                              self.p_keep_ff)
        self.hidden_output_2 = tf.contrib.layers.fully_connected(self.hidden_1_dropped,
                                                                 self.ff_size,
                                                                 tf.tanh)
        self.hidden_2_dropped = tf.nn.dropout(self.hidden_output_2,
                                              self.p_keep_ff)
        self.hidden_output_3 = tf.contrib.layers.fully_connected(self.hidden_2_dropped,
                                                                 self.ff_size,
                                                                 tf.tanh)
        self.hidden_3_dropped = tf.nn.dropout(self.hidden_output_3,
                                              self.p_keep_ff)
        self.logits = tf.contrib.layers.fully_connected(inputs=self.hidden_3_dropped,
                                                        num_outputs=3,
                                                        activation_fn=None)
        return self.logits

    @define_scope
    def loss(self):
        return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.y,
                                                                     logits=self.logits,
                                                                     name='loss'))

    @define_scope
    def optimize(self):
        #return tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        weights = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('weights:0')]
        grads_and_vars = optimizer.compute_gradients(self.loss, weights)
        clipped_grads_and_vars = clip_gradients(grads_and_vars, self.grad_clip_norm)
        return optimizer.apply_gradients(clipped_grads_and_vars)


class Aligned(Model):
    def __init__(self, word_embed_length=300, learning_rate=0.001, hidden_size=100,
                 p_keep_input=0.8, p_keep_hidden=0.5, grad_norm=3.0):
        Model.__init__(self, word_embed_length, learning_rate, hidden_size,
                       p_keep_input, p_keep_hidden, grad_norm)
        self.name = 'aligned'
        self._bi_rnns

    @define_scope('alignment')
    def _alignment(self):
        pass

    @define_scope('bi_rnns')
    def _bi_rnns(self):
        _, self.premise_output_states = bi_rnn(self.premises, self.hidden_size, 'premise_bi_rnn')
        self.premise_out = tf.concat([state.c for state in self.premise_output_states], axis=1)
        _, self.hypothesis_output_states = bi_rnn(self.hypotheses, self.hidden_size, 'hypothesis_bi_rnn')
        self.hypothesis_out = tf.concat([state.c for state in self.hypothesis_output_states], axis=1)
        self.rnn_output = tf.concat([self.premise_out, self.hypothesis_out],
                                    axis=1,
                                    name='concatenated_sentences')
        return self.rnn_output  # batch_size x (4 * hidden_size)


if __name__ == '__main__':
    collection = 'dev'
    batch_gen = get_batch_gen('dev')
    learning_rate = 1e-3
    model = LSTMEncoder(learning_rate=learning_rate)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run(model.premise_reduced_encoding,
                          feed_dict(model, next(batch_gen)))
        print(result.shape)
