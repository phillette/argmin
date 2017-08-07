"""Trainer class for TensorFlow models."""
import tensorflow as tf
import os
from coldnet import training


class TensorFlowTrainer(training.TrainerBase):
    """Base class for training TensorFlow models."""

    def __init__(self, model, history, train_data, tune_data, ckpt_dir):
        super(TensorFlowTrainer, self).__init__(
            model, history, train_data, tune_data)
        self.ckpt_dir = ckpt_dir
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(model.embeddings_init,
                      feed_dict={model.embeddings_ph: model.embedding_matrix})

    def _checkpoint(self, is_best):
        path = training.model_path(self.ckpt_dir, self.name, is_best)
        self.saver.save(
            self.sess,
            path,
            global_step=self.history.global_step)

    def _load_last(self):
        path = training.model_path(self.ckpt_dir, self.name, False)
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(path))
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise Exception('Checkpoint "%s" not found' % path)

    def predict(self, batch):
        feed_dict = self.model.dropout_feed_dict()
        feed_dict[self.model.premise_ixs] = batch.premises
        feed_dict[self.model.hypothesis_ixs] = batch.hypotheses
        feed_dict[self.model.labels] = batch.labels
        acc = self.sess.run(self.model.accuracy, feed_dict=feed_dict)
        return acc


class FeedDictTrainer(TensorFlowTrainer):
    """For training TensorFlow models with a feed dict."""

    def __init__(self, model, history, train_batcher, tune_batcher, ckpt_dir):
        super(FeedDictTrainer, self).__init__(
            model, history, train_batcher, tune_batcher, ckpt_dir)

    def step(self, batch):
        feed_dict = self.model.dropout_feed_dict()
        feed_dict[self.model.premise_ixs] = batch.premises
        feed_dict[self.model.hypothesis_ixs] = batch.hypotheses
        feed_dict[self.model.labels] = batch.labels
        loss, acc, _ = self.sess.run([self.model.loss,
                                      self.model.accuracy,
                                      self.model.optimize],
                                     feed_dict)
        return loss, acc
