import unittest
from align1 import *
import tensorflow as tf
import numpy as np


class Align1Tests(unittest.TestCase):
    def setUp(self):
        pass

    def test_sentence_rnn_output(self):
        premises, hypotheses = data()
        sentences = np.random.randn(3, 8, 300)  # at the moment it's like this - pre-padding out per batch
        output, initial_state, output_state = create_sentence_rnn(premises, 'premises')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            result = sess.run(output, {premises: sentences})
            print(type(result))
            print(result.shape)


if __name__ == '__main__':
    unittest.main()
