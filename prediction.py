import os
from batching import get_batch_gen, NUM_ITERS, ENCODING_TO_LABEL, COLLECTION_SIZE
import numpy as np
from util import load_checkpoint, feed_dict, feed_dict2
import tensorflow as tf
import itertools
from mongoi import get_repository
import pandas as pd
from aligned import Alignment
from model_base import Config


def accuracy(model, db, collection, sess, transfer=False):
    # make sure sess.run(tf.global_variables_initializer()) has already been called
    batch_gen = get_batch_gen(db, collection)
    num_iters = NUM_ITERS[db][collection]
    saver = tf.train.Saver()
    load_checkpoint(model, saver, sess, transfer)
    average_accuracy = 0
    for iter in range(num_iters):
        batch = next(batch_gen)
        batch_accuracy = sess.run(model.accuracy,
                                  feed_dict(model,
                                            batch))
        average_accuracy += batch_accuracy
    print('%s %s set accuracy = %s' % (db, collection, average_accuracy / num_iters))


def accuracy2(model, batch_gen, num_iters, sess):
    average_accuracy = 0
    for iter in range(num_iters):
        batch = next(batch_gen)
        batch_accuracy = sess.run(model.accuracy, feed_dict2(model, batch))
        average_accuracy += batch_accuracy
    print('Accuracy: %s' % (average_accuracy / num_iters))


def evaluate(model, sess, db='snli', collection='test', transfer=False):
    batch_gen = get_batch_gen(db, collection)
    num_iters = NUM_ITERS[db][collection]
    load_checkpoint(model, tf.train.Saver(), sess, transfer)
    re = ResultExaminer(db, collection)
    for iter in range(num_iters):
        batch = next(batch_gen)
        predicted_labels, confidences, correct_predictions = sess.run([model.predicted_labels,
                                                                       model.confidences,
                                                                       model.correct_predictions],
                                                                      feed_dict(model, batch))
        re.new_batch_results(batch.ids, predicted_labels, confidences, correct_predictions)
    return re


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


class ErrorExaminer:
    def __init__(self, db, collection, correct_ids):
        self.db = db
        self.collection = collection
        self._repository = get_repository(db, collection)
        self.collection_size = COLLECTION_SIZE[db][collection]
        self.correct_ids = correct_ids
        self.random_correct = []
        self.random_incorrect = []
        self.reset_random_pool()

    def random_correct(self):
        id = self._random_correct_id()
        doc = self._repository.get(id)
        self._print_doc(doc, True)

    def random_incorrect(self):
        id = self._random_incorrect_id()
        doc = self._repository.get(id)
        self._print_doc(doc, False)

    def reset_random_pool(self):
        self.random_correct = [id for id in self.correct_ids]
        self.random_incorrect = [id for id in list(range(self.collection_size)) if id in self.correct_ids]

    def _random_correct_id(self):
        next_id = np.random.choice(self.random_correct, 1)
        self.random_correct.remove(next_id)
        return next_id

    def _random_incorrect_id(self):
        next_id = np.random.choice(self.random_incorrect, 1)
        self.random_incorrect.remove(next_id)
        return next_id

    def _print_doc(self, doc, correct):
        if correct:
            print('CORRECT classification:')
        else:
            print('INCORRECT classification:')
        print('Gold Label: %s' % doc['gold_label'])
        print('Predicted Label: %s' % None)
        print('Premise:')
        print(doc['sentence1'])
        print('Hypothesis:')
        print(doc['sentence2'])


class ResultExaminer:
    def __init__(self, db, collection):
        self.n = COLLECTION_SIZE[db][collection]
        self.results = pd.DataFrame(data=None,
                                    index=np.arange(self.n) + 1,
                                    columns=['predicted_label', 'confidence', 'correct'])

    def new_batch_results(self, ids, predicted_labels, confidences, correct_predictions):
        self.results.iloc[ids] = [predicted_labels, confidences, correct_predictions]


if __name__ == '__main__':
    config = Config(learning_rate=1e-4,
                    p_keep_rnn=1.0,
                    p_keep_input=1.0,
                    p_keep_ff=1.0,
                    grad_clip_norm=5.0,
                    lamda=0.0)
    model = Alignment(config, 100)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        re = evaluate(model, sess)
