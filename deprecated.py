import tensorflow as tf
from tf_decorators import define_scope
from util import dropout_vector, add_bias, clip_gradients


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