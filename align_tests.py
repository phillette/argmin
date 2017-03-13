import unittest
from align1 import *
import tensorflow as tf
import numpy as np


class Align1Tests(unittest.TestCase):
    def setUp(self):
        self.premises, self.hypotheses = data()
        self.premises_input = np.random.randn(3, 50, 300)
        self.hypotheses_input = np.random.rand(3, 50, 300)
        self.feed_dict = {self.premises: self.premises_input,
                          self.hypotheses: self.hypotheses_input}
        self.premises_output, self.hypotheses_output = create_rnns(self.premises,
                                                                   self.hypotheses)
        self.encoded_premises = encode_sentences(self.premises_output,
                                                 'encoded_premises')
        self.encoded_hypotheses = encode_sentences(self.hypotheses_output,
                                                   'encoded_hypotheses')
        self.concatenation = classify(self.encoded_premises,
                                      self.encoded_hypotheses)
        self.sess = tf.Session()

    def test_rnn_output(self):
        p = self.sess.run(self.premises_output, self.feed_dict)
        h = self.sess.run(self.hypotheses_output, self.feed_dict)
        print(p.shape)
        print(h.shape)

    def tearDown(self):
        self.sess.close()


if __name__ == '__main__':
    unittest.main()
