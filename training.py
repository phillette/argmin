import tensorflow as tf
import numpy as np
import batching
import stats
import util
import time
import datetime
import matplotlib.pyplot as plt


class History:
    def __init__(self):
        self.iter = []
        self.loss = []
        self.accuracy = []

    def report(self, iter, loss, accuracy):
        self.iter.append(iter)
        self.loss.append(loss)
        self.accuracy.append(accuracy)

    def visualize(self):
        plt.plot(self.iter, self.loss, 'r-')
        plt.plot(self.iter, self.accuracy, 'b-')
        plt.show()


def train(model, db, collection, num_epochs, sess, batch_size=4,
          load_ckpt=True, save_ckpt=True, transfer=False, summarise=True):
    # make sure sess.run(tf.global_variables_initializer() has already been run)
    writer = tf.summary.FileWriter(util.log_graph_path(model.name), sess.graph)
    saver = tf.train.Saver()
    if load_ckpt:
        util.load_checkpoint(model, saver, sess, transfer)
    model.in_training = True
    start_reported = False
    average_loss = 0.0
    average_accuracy = 0.0
    starting_iter = model.global_step.eval()
    iter = starting_iter
    num_iters = batching.num_iters(db, collection, batch_size)
    epoch_time_takens = []
    iter_time_takens = []
    history = History()
    start = time.time()
    for epoch in range(num_epochs):
        print('Epoch %s' % (epoch + 1))
        epoch_start = time.time()
        batch_gen = batching.get_batch_gen(db, collection, batch_size=batch_size)
        last_iter = starting_iter + ((epoch + 1) * num_iters)
        while iter <= last_iter:  # check that <= is right, I had < before
            iter_start = time.time()
            batch = next(batch_gen)
            batch_loss, batch_accuracy, _, summary = sess.run([model.loss,
                                                               model.accuracy,
                                                               model.optimize,
                                                               model.summary],
                                                              util.feed_dict(model, batch))
            if not start_reported:
                print('Starting condition: loss = %s; accuracy = %s' % (batch_loss, batch_accuracy))
                start_reported = True
            average_loss += batch_loss
            average_accuracy += batch_accuracy
            history.report(iter, average_loss, average_accuracy)
            iter_end = time.time()
            iter_time_taken = iter_end - iter_start
            iter_time_takens.append(iter_time_taken)
            if summarise:
                writer.add_summary(summary, global_step=model.global_step.eval())
            iter += 1
            average_time = np.average(iter_time_takens)
            iters_remaining = last_iter - iter
            if iter % stats.REPORT_EVERY[db][collection] == 0:
                print('Step %s (%s%%): '
                      'loss = %s; '
                      'accuracy = %s; '
                      'time = %s; '
                      'remaining = %s' % (iter,
                                          iter / last_iter,
                                          average_loss / iter,
                                          average_accuracy / iter,
                                          average_time,
                                          average_time * iters_remaining))
                if save_ckpt:
                    util.save_checkpoint(model, saver, sess, transfer, iter)
        epoch_end = time.time()
        epoch_time_taken = epoch_end - epoch_start
        epoch_time_takens.append(epoch_time_taken)
        print('Epoch time taken: %s; average = %s; ETC: %s' % (epoch_time_taken,
                                                               np.average(epoch_time_takens),
                                                               start + np.average(epoch_time_takens) * num_epochs))
    writer.close()
    model.in_training = False
    history.visualize()
    util.save_pickle(history, 'histories/%s_%s' % (model.name,
                                                   datetime.datetime.now().strftime('%d-%b-%Y_%H-%M-%S')))
