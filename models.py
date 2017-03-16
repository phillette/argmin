import tensorflow as tf
from tf_decorators import define_scope
from training import train
from process_data import get_batch_gen


class AdditiveSentence:
    def __init__(self, hidden_size=300, alpha=0.001):
        self.name = 'additive_sentence'
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.premises = tf.placeholder(tf.float64, [None, None, None], name='premises')
        self.hypotheses = tf.placeholder(tf.float64, [None, None, None], name='hypotheses')
        self.y = tf.placeholder(tf.float64, [None, 1, 3], name='y')

    @define_scope
    def logits(self):
        premise_vectors = tf.reduce_sum(self.premises, 1)  #    [batch_size, max_premise_length, word_embed_length]
                                                           # => [batch_size, word_embed_length]
        hypothesis_vectors = tf.reduce_sum(self.hypotheses, 1)  # as above
        concatenated_sents = tf.concat([premise_vectors, hypothesis_vectors], 1)  # [1, 2 * word_embed_length]
        hidden_output_1 = tf.contrib.layers.fully_connected(concatenated_sents,
                                                            tf.shape(concatenated_sents, 1),
                                                            tf.tanh,
                                                            name='hidden_layer_1')
        hidden_output_2 = tf.contrib.layers.fully_connected(hidden_output_1,
                                                            tf.shape(concatenated_sents, 1),
                                                            tf.tanh,
                                                            name='hidden_layer_2')
        logits = tf.contrib.layers.fully_connected(hidden_output_2,
                                                   tf.shape(concatenated_sents, 1),
                                                   None,
                                                   name='logits')
        return logits

    @define_scope
    def loss(self):
        return tf.nn.softmax_cross_entropy_with_logits(self.logits, self.y)

    @define_scope
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(self.alpha)
        return optimizer.minimize(self.loss)

    @define_scope()
    def predict(self):
        return tf.nn.softmax(self.logits)


class HLastSentence:
    pass


class LearnedSentence:
    pass


class Aligned:
    pass


if __name__ == '__main__':
    batch_size = 100
    model = AdditiveSentence()
    batch_gen = get_batch_gen(batch_size, 'dev')
    train(model, batch_gen)
