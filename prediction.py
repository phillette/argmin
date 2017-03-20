import tensorflow as tf
import os
from process_data import get_batch_gen, ITER_COUNTS, ENCODING_TO_LABEL
from models import AdditiveSentence
import numpy as np


def accuracy(model, collection):
    batch_gen = get_batch_gen(collection)
    num_iters = ITER_COUNTS[collection]
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/%s/%s.ckpt' % (model.name,
                                                                                         model.name)))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise Exception('No checkpoint found')
        average_accuracy = 0
        for iter in range(num_iters):
            batch = next(batch_gen)
            batch_accuracy = sess.run(model.accuracy, {model.premises: batch.premises,
                                                       model.hypotheses: batch.hypotheses,
                                                       model.y: batch.labels})
            print('Batch %s accuracy = %s' % (iter, batch_accuracy))
            average_accuracy += batch_accuracy
        print('Final accuracy = %s' % (average_accuracy / num_iters))


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
    model = AdditiveSentence()
    dev_acc = accuracy(model, 'dev')
