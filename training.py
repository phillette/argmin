import tensorflow as tf
import numpy as np
import batching
import stats
import util
import time


def train(model, db, collection, num_epochs, sess, load_ckpt=True, save_ckpt=True, transfer=False, summarise=True):
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
    epoch_time_takens = []
    iter_time_takens = []
    start = time.time()
    for epoch in range(num_epochs):
        print('Epoch %s' % (epoch + 1))
        epoch_start = time.time()
        batch_gen = batching.get_batch_gen(db, collection)
        while iter < starting_iter + (epoch + 1) * stats.NUM_ITERS[db][collection]:
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
            iter_end = time.time()
            iter_time_taken = iter_end - iter_start
            iter_time_takens.append(iter_time_taken)
            if summarise:
                writer.add_summary(summary, global_step=model.global_step.eval())
            iter += 1
            if iter % stats.REPORT_EVERY[db][collection] == 0:
                print('Step %s: loss = %s; accuracy = %s; time = %s' % (iter,
                                                                        average_loss / iter,
                                                                        average_accuracy / iter,
                                                                        np.average(iter_time_takens)))
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
