import tensorflow as tf
from tensorflow.contrib import rnn
from tf_decorators import define_scope
from training import train
from process_data import BATCH_SIZE, LONGEST_SENTENCE_SNLI
from util import add_bias, dropout_vector, clip_gradients, length


def bi_rnn(sentences, hidden_size, scope,
           dropout=False, p_dropout=0.8):
    sequence_lengths = length(sentences)
    forward_cell = rnn.BasicLSTMCell(hidden_size, forget_bias=1.0)
    backward_cell = rnn.BasicLSTMCell(hidden_size, forget_bias=1.0)
    if dropout:
        forward_cell = tf.contrib.rnn.DropoutWrapper(cell=forward_cell,
                                                     output_keep_prob=p_dropout)
        backward_cell = tf.contrib.rnn.DropoutWrapper(cell=backward_cell,
                                                      output_keep_prob=p_dropout)
    output, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=forward_cell,
                                                            cell_bw=backward_cell,
                                                            inputs=sentences,
                                                            sequence_length=sequence_lengths,
                                                            dtype=tf.float64,
                                                            scope=scope)
    return output, output_states


class ModelBase:
    def __init__(self, word_embed_length=300, learning_rate=0.001, hidden_size=100):
        self.word_embed_length = word_embed_length
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.time_steps = LONGEST_SENTENCE_SNLI
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
    def logits(self):
        raise NotImplementedError()

    @define_scope
    def loss(self):
        return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.y,
                                                                     logits=self.logits,
                                                                     name='loss'))

    @define_scope
    def optimize(self):
        return tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


class AdditiveSentence:
    def __init__(self, word_embed_length=300, learning_rate=0.001,
                 p_keep_input=0.8, p_keep_hidden=0.5):
        self.name = 'additive_sentence'
        self.word_embed_length = word_embed_length
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, dtype=tf.int32,
                                       trainable=False,
                                       name='global_step')
        self.p_keep_input = tf.Variable(p_keep_input, dtype=tf.float64, trainable=False)
        self.p_keep_hidden = tf.Variable(p_keep_hidden, dtype=tf.float64, trainable=False)
        self.concatenated_length = 600
        self.hidden_layer_size = 1200
        self.drop_input = dropout_vector(self.p_keep_input.initialized_value(),
                                         [1, self.concatenated_length + 1])
        self.drop_hidden = dropout_vector(self.p_keep_hidden.initialized_value(),
                                          [1, self.hidden_layer_size + 1])
        self._data
        self._concatenated_sents
        self._parameters
        self._feedforward_train
        self._feedforward_test
        self.loss
        self.predict
        self.accuracy
        self.accuracy_train
        self.optimize

    @define_scope
    def accuracy(self):
        correct_predictions = tf.equal(tf.argmax(self.predict, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float64))
        return accuracy

    @define_scope('accuracy_train')
    def accuracy_train(self):
        correct_predictions = tf.equal(tf.argmax(self._feedforward_train, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float64))
        return accuracy

    @define_scope('concatenated_sents')
    def _concatenated_sents(self):
        premise_vectors = tf.reduce_sum(self.premises, 1)  # [batch_size, word_embed_length]
        hypothesis_vectors = tf.reduce_sum(self.hypotheses, 1)  # [batch_size, word_embed_length]
        concatenated_sents = tf.concat([premise_vectors, hypothesis_vectors],
                                       1,
                                       name='concatenated_sentences')  # [1, concatenated_length]
        return concatenated_sents

    @define_scope('data')
    def _data(self):
        self.premises = tf.placeholder(dtype=tf.float64,
                                       shape=[None, None, self.word_embed_length],
                                       name='premises')
        self.hypotheses = tf.placeholder(dtype=tf.float64,
                                         shape=[None, None, self.word_embed_length],
                                         name='hypotheses')
        self.y = tf.placeholder(dtype=tf.float64,
                                shape=[None, 3],
                                name='y')

    @define_scope('feedforward_test')
    def _feedforward_test(self):
        input_with_bias = add_bias(self._concatenated_sents)
        hidden_output_1 = tf.tanh(tf.matmul(input_with_bias,
                                            tf.multiply(self.Theta1, self.p_keep_input)),
                                  name='augmented_hidden_output_1')
        hidden_output_1_with_bias = add_bias(hidden_output_1)
        hidden_output_2 = tf.tanh(tf.matmul(hidden_output_1_with_bias,
                                            tf.multiply(self.Theta2, self.p_keep_hidden)),
                                  name='augmented_hidden_output_2')
        hidden_output_2_with_bias = add_bias(hidden_output_2)
        logits = tf.matmul(hidden_output_2_with_bias,
                           tf.multiply(self.Theta3, self.p_keep_hidden),
                           name='augmented_logits')
        return logits

    @define_scope('feedforward_train')
    def _feedforward_train(self):
        input_with_bias = add_bias(self._concatenated_sents)
        dropped_input = tf.multiply(input_with_bias,
                                    self.drop_input,
                                    name='dropped_input')
        hidden_output_1 = tf.tanh(tf.matmul(dropped_input,
                                            self.Theta1),
                                  name='hidden_output_1')
        hidden_output_1_with_bias = add_bias(hidden_output_1)
        dropped_hidden_1 = tf.multiply(hidden_output_1_with_bias,
                                       self.drop_hidden,
                                       name='dropped_hidden_1')
        hidden_output_2 = tf.tanh(tf.matmul(dropped_hidden_1,
                                            self.Theta2),
                                  name='hidden_output_2')
        hidden_output_2_with_bias = add_bias(hidden_output_2)
        dropped_hidden_2 = tf.multiply(hidden_output_2_with_bias,
                                       self.drop_hidden,
                                       name='dropped_hidden_2')
        logits = tf.matmul(dropped_hidden_2, self.Theta3)
        return logits

    @define_scope
    def loss(self):
        return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self._feedforward_train,
                                                                     labels=self.y))

    @define_scope
    def optimize(self):
        optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
        #optimizer = tf.train.AdamOptimizer(self.learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.loss, self._parameters)
        clipped_grads_and_vars = clip_gradients(grads_and_vars)
        return optimizer.apply_gradients(clipped_grads_and_vars)

    @define_scope('parameters')
    def _parameters(self):
        self.Theta1 = tf.Variable(tf.random_uniform(shape=[self.concatenated_length + 1,
                                                           self.hidden_layer_size],
                                                    minval=-1.0,
                                                    maxval=1.0,
                                                    dtype=tf.float64),
                                  name='Theta1')
        self.Theta2 = tf.Variable(tf.random_uniform(shape=[self.hidden_layer_size + 1,
                                                           self.hidden_layer_size],
                                                    minval=-1.0,
                                                    maxval=1.0,
                                                    dtype=tf.float64),
                                  name='Theta2')
        self.Theta3 = tf.Variable(tf.random_uniform(shape=[self.hidden_layer_size + 1,
                                                           3],  # the number of labels
                                                    minval=-1.0,
                                                    maxval=1.0,
                                                    dtype=tf.float64),
                                  name='Theta3')
        return [self.Theta1, self.Theta2, self.Theta3]

    @define_scope
    def predict(self):
        return tf.nn.softmax(self._feedforward_test)


class BiRNN:
    """
    Learning rate: 1e-3
    Training set accuracy = 0.969
    Test set accuracy = 0.6828
    Carstens = 0.46 (not fully trained but best average accuracy of a batch before gradients explode)
    """
    def __init__(self, word_embed_length=300, learning_rate=0.001, hidden_size=100,
                 dropout=False, p_dropout=0.8):
        self.name = 'bi_rnn'
        self.word_embed_length = word_embed_length
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.p_dropout = p_dropout
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
        _, self.premise_output_states = bi_rnn(self.premises, self.hidden_size, 'premise_bi_rnn',
                                               self.dropout, self.p_dropout)
        self.premise_out = tf.concat([state.c for state in self.premise_output_states], axis=1)
        _, self.hypothesis_output_states = bi_rnn(self.hypotheses, self.hidden_size, 'hypothesis_bi_rnn',
                                                  self.dropout, self.p_dropout)
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
                                                                 self.hidden_size,
                                                                 tf.tanh)
        self.hidden_output_2 = tf.contrib.layers.fully_connected(self.hidden_output_1,
                                                                 self.hidden_size,
                                                                 tf.tanh)
        self.hidden_output_3 = tf.contrib.layers.fully_connected(self.hidden_output_2,
                                                                 self.hidden_size,
                                                                 tf.tanh)
        self.logits = tf.contrib.layers.fully_connected(inputs=self.hidden_output_3,
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
        return tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


class BiRNNDropout(ModelBase):
    def __init__(self, word_embed_length=300, learning_rate=0.001, hidden_size=100):
        ModelBase.__init__(self, word_embed_length=300, learning_rate=0.001, hidden_size=100)
        self.name = 'bi_rnn'

        self._data
        self._bi_rnns
        self._logits
        self.loss
        self.optimize
        self.accuracy_train
        self.accuracy




class Aligned(ModelBase):
    def __init__(self):
        ModelBase.__init__(self, word_embed_length=300, learning_rate=0.001, hidden_size=100)
        self.name = 'aligned'
        self._bi_rnns

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
    collection = 'train'
    num_epochs = 10
    learning_rate = 1e-3
    model = BiRNN(learning_rate=learning_rate)
    train(model, collection, num_epochs)
