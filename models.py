import tensorflow as tf
from tf_decorators import define_scope
from training import train
from process_data import get_batch_gen


class AdditiveSentence:
    def __init__(self, word_embed_length=300, hidden_size=300, alpha=0.001):
        self.name = 'additive_sentence'
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.premises = tf.placeholder(dtype=tf.float64, shape=[None, None, word_embed_length], name='premises')
        self.hypotheses = tf.placeholder(dtype=tf.float64, shape=[None, None, word_embed_length], name='hypotheses')
        self.concatenated_length = 2 * word_embed_length
        self.y = tf.placeholder(dtype=tf.float64, shape=[None, 3], name='y')
        self._concatenated_sents
        self._hidden_layer_1
        self._hidden_layer_2
        self._logits
        self.loss
        self.predict
        self.optimize
        self._correct_predictions
        self.accuracy

    @define_scope
    def accuracy(self):
        accuracy = tf.reduce_mean(tf.cast(self._correct_predictions, tf.float64))
        return accuracy

    @define_scope('concatenated_sents')
    def _concatenated_sents(self):
        premise_vectors = tf.reduce_sum(self.premises, 1)  # [batch_size, word_embed_length]
        hypothesis_vectors = tf.reduce_sum(self.hypotheses, 1)  # [batch_size, word_embed_length]
        concatenated_sents = tf.concat([premise_vectors, hypothesis_vectors], 1)  # [1, concatenated_length]
        return concatenated_sents

    @define_scope('correct_predictions')
    def _correct_predictions(self):
        return tf.equal(tf.argmax(self.predict, 1), tf.argmax(self.y, 1))

    @define_scope('hidden_layer_1')
    def _hidden_layer_1(self):
        hidden_output_1 = tf.contrib.layers.fully_connected(inputs=self._concatenated_sents,
                                                            num_outputs=600,
                                                            activation_fn=tf.tanh)
        return hidden_output_1

    @define_scope('hidden_layer_2')
    def _hidden_layer_2(self):
        hidden_output_2 = tf.contrib.layers.fully_connected(inputs=self._hidden_layer_1,
                                                            num_outputs=600,
                                                            activation_fn=tf.tanh)
        return hidden_output_2

    @define_scope('logits')
    def _logits(self):
        logits = tf.contrib.layers.fully_connected(inputs=self._hidden_layer_2,
                                                   num_outputs=3,
                                                   activation_fn=None)
        return logits

    @define_scope
    def loss(self):
        return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self._logits, labels=self.y))

    @define_scope
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(self.alpha)
        return optimizer.minimize(self.loss)

    @define_scope
    def predict(self):
        return tf.nn.softmax(self._logits)


class HLastSentence:
    pass


class LearnedSentence:
    pass


class Aligned:
    pass


if __name__ == '__main__':
    batch_size = 10
    model = AdditiveSentence(alpha=0.01)  # i think the optimizer could be moved to the training function
    for i in range(10):
        print('Epoch %s' % (i + 1))
        batch_gen = get_batch_gen(batch_size, 'dev')  # somehow this needs to be randomized for each epoch
        train(model, batch_gen)
