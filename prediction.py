import tensorflow as tf
import os
from process_data import get_batch_gen, NUM_ITERS, ENCODING_TO_LABEL, BATCH_SIZE
from models import *
import numpy as np
from util import load_checkpoint, feed_dict


def accuracy(model, collection, sess):
    # make sure sess.run(tf.global_variables_initializer()) has already been called
    batch_gen = get_batch_gen(collection)
    num_iters = NUM_ITERS[collection]
    saver = tf.train.Saver()
    load_checkpoint(model, saver, sess)
    average_accuracy = 0
    for iter in range(num_iters):
        batch = next(batch_gen)
        batch_accuracy = sess.run(model.accuracy, feed_dict(model, batch))
        average_accuracy += batch_accuracy
    print('%s set accuracy = %s' % (collection, average_accuracy / num_iters))


class Prediction:
    def __init__(self, encoding, confidence):
        self.encoding = encoding
        self.confidence = confidence

    def label(self, threshold=None):
        if threshold:
            return ENCODING_TO_LABEL[self.encoding] if self.confidence > threshold else ENCODING_TO_LABEL[0]
        return ENCODING_TO_LABEL[self.encoding]


def predict(model, premise, hypothesis):
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/%s/%s.ckpt' % (model.name,
                                                                                         model.name)))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise Exception('No checkpoint found')
        probabilities = sess.run(model.predict, {model.premises: premise,
                                                 model.hypotheses: hypothesis})  # shouldn't need y (?)
        return Prediction(np.argmax(probabilities, 1)[0], np.max(probabilities))


if __name__ == '__main__':
    collection = 'dev'
    model = BiRNN(BATCH_SIZE[collection])
    accuracy(model, collection)
