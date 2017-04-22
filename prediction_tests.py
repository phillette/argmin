import unittest
import numpy as np
from prediction import *
import tensorflow as tf
from model_base import Model
from decorators import define_scope


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        self.sess = tf.Session()

    def tearDown(self):
        self.sess.close()


class FakeBatch:
    # can't pass it in the way the function is written currently...
    # testing as I go really is the way to ensure my code is unit test friendly... but I knew that already...
    pass


class FakeModel(Model):
    def __init__(self, config):
        Model.__init__(self, config)
        self.logits
        self.predicted_labels
        self.confidences
        self.correct_predictions

    @define_scope
    def logits(self):
        return tf.Variable(np.array([[0.1, 0.4, 0.5],
                                     [0.4, 0.1, 0.5],
                                     [0.5, 0.4, 0.1]]))


if __name__ == '__main__':
    unittest.main()
