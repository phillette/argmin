import unittest
from align1 import *
import tensorflow as tf
import numpy as np


def get_batch_gen():
    X_p = list(np.random.randn(5000, LONGEST_SENTENCE, WORD_EMBED_DIM))
    X_h = list(np.random.randn(5000, LONGEST_SENTENCE, WORD_EMBED_DIM))
    Y = np.zeros((5000, 1, 3))
    Y[np.arange(len(Y)), 0, np.random.randint(3, size=(5000,))] = 1
    for ndx in range(0, 5000, BATCH_SIZE):
        yield X_p[ndx:min(ndx + BATCH_SIZE, 5000)], \
              X_h[ndx:min(ndx + BATCH_SIZE, 5000)], \
              Y[ndx:min(ndx + BATCH_SIZE, 5000)]


class Align1Tests(unittest.TestCase):
    def setUp(self):
        self.premises_input = np.random.randn(BATCH_SIZE, LONGEST_SENTENCE, WORD_EMBED_DIM)
        self.hypotheses_input = np.random.rand(3, 50, 300)
        self.y_input = np.array([[[1., 0., 0.]],
                                [[0., 1., 0.]],
                                [[0., 0., 1.]]])
        self.premises, self.hypotheses, self.y = data()
        self.feed_dict = {self.premises: self.premises_input,
                          self.hypotheses: self.hypotheses_input,
                          self.y: self.y_input}
        self.premises_output, self.hypotheses_output = create_rnns(self.premises,
                                                                   self.hypotheses)
        self.encoded_premises = encode_sentences(self.premises_output,
                                                 'encoded_premises')
        self.encoded_hypotheses = encode_sentences(self.hypotheses_output,
                                                   'encoded_hypotheses')
        self.logits = classify(self.encoded_premises, self.encoded_hypotheses)
        self.cost = loss(self.logits, self.y)
        self.optimizer = optimization(self.cost)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    #def test_rnn_output(self):
    #    p = self.sess.run(self.premises_output, self.feed_dict)
    #    h = self.sess.run(self.hypotheses_output, self.feed_dict)
    #    assert p.shape == (BATCH_SIZE, LONGEST_SENTENCE, WORD_EMBED_DIM)
    #    assert h.shape == (BATCH_SIZE, LONGEST_SENTENCE, WORD_EMBED_DIM)

    #def test_encoding(self):
    #    result = self.sess.run(self.encoded_premises, self.feed_dict)
    #    print(result.shape)
    #    print(result)

    #def test_classify(self):
    #    result = self.sess.run(self.y_hat, self.feed_dict)
    #    print(result.shape)

    #def test_cost(self):
    #    result = self.sess.run(self.cost, self.feed_dict)
    #    print(result)

    #def test_optimization(self):
    #    loss, _ = self.sess.run([self.cost, self.optimizer], self.feed_dict)
    #    print(loss)

    def test_train(self):
        average_loss = 0.0
        step = 0
        for premises, hypotheses, labels in get_batch_gen():
            step += 1
            batch_loss, _ = self.sess.run([self.cost, self.optimizer],
                                          {
                                              self.premises: premises,
                                              self.hypotheses: hypotheses,
                                              self.y: labels
                                          })
            average_loss += batch_loss
            print('Average loss at step %s: %s' % (step, average_loss))

    def tearDown(self):
        self.sess.close()


if __name__ == '__main__':
    unittest.main()
