import os
import batching
import numpy as np
import util
import tensorflow as tf
import mongoi
import pandas as pd
import stats
import labeling


def load_ckpt_at_epoch(model, epoch, db_name, collection_name,
                       batch_size, subset_size, sess, saver):
    iters_per_epoch = batching.num_iters(db_name,
                                         collection_name,
                                         batch_size,
                                         subset_size)
    global_step = util.scale_epoch_to_iter(epoch, iters_per_epoch)
    util.load_checkpoint_at_step(model.name, global_step, saver, sess)


def accuracy(model,
             db,
             collection,
             sess,
             batch_size=32,
             subset_size=None,
             surpress_print=False,
             batch_gen_fn=batching.get_batch_gen,
             feed_dict_fn=util.feed_dict):
    # make sure sess.run(tf.global_variables_initializer()) has been called
    # make sure the checkpoint is loaded too if necessary
    batch_gen = batch_gen_fn(db,
                             collection,
                             batch_size=batch_size)
    num_iters = batching.num_iters(db=db,
                                   collection=collection,
                                   batch_size=batch_size,
                                   subset_size=subset_size)
    accumulated_accuracy = 0.0
    for iter in range(num_iters):
        batch = next(batch_gen)
        batch_accuracy = sess.run(model.accuracy,
                                  feed_dict_fn(model,
                                               batch))
        accumulated_accuracy += batch_accuracy
    if not surpress_print:
        print('%s %s set accuracy = %s%%'
              % (db,
                 collection,
                 round(accumulated_accuracy / num_iters * 100, 2)))
    return accumulated_accuracy / num_iters


def evaluate(model, sess, db='snli', collection='test', transfer=False):
    batch_gen = batching.get_batch_gen(db, collection)
    num_iters = batching.num_iters(db, collection)
    util.load_checkpoint(model, tf.train.Saver(), sess, transfer)
    re = ResultExaminer(db, collection)
    for iter in range(num_iters):
        batch = next(batch_gen)
        predicted_labels, confidences, correct_predictions \
            = sess.run([model.predicted_labels,
                        model.confidences,
                        model.correct_predictions],
                        util.feed_dict(model, batch))
        re.new_batch_results(batch.ids,
                             predicted_labels,
                             confidences,
                             correct_predictions)
    return re


class Prediction:
    def __init__(self, encoding, confidence):
        self.encoding = encoding
        self.confidence = confidence

    def label(self, threshold=None):
        if threshold:
            return labeling.ENCODING_TO_LABEL[self.encoding] \
                if self.confidence > threshold \
                else labeling.ENCODING_TO_LABEL[0]
        return labeling.ENCODING_TO_LABEL[self.encoding]


def predict(model, premise, hypothesis):
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(
            os.path.dirname('checkpoints/%s/%s.ckpt' % (model.name,
                                                        model.name)))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise Exception('No checkpoint found')
        probabilities = sess.run(model.predict, {model.premises: premise,
                                                 model.hypotheses: hypothesis})
        return Prediction(np.argmax(probabilities, 1)[0],
                          np.max(probabilities))


class ErrorExaminer:
    def __init__(self, db, collection, correct_ids):
        self.db = db
        self.collection = collection
        self._repository = mongoi.get_repository(db, collection)
        self.collection_size = stats.COLLECTION_SIZE[db][collection]
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
        self.random_incorrect \
            = [id for
               id in list(range(self.collection_size)) if
               id in self.correct_ids]

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
        self.n = stats.COLLECTION_SIZE[db][collection]
        self.results = pd.DataFrame(data=None,
                                    index=np.arange(self.n) + 1,
                                    columns=['predicted_label',
                                             'confidence',
                                             'correct'])

    def new_batch_results(self,
                          ids,
                          predicted_labels,
                          confidences,
                          correct_predictions):
        self.results.iloc[ids] = [predicted_labels,
                                  confidences,
                                  correct_predictions]
