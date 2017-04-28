import tensorflow as tf
import numpy as np
import batching
import util
import stats
import time
import datetime
import matplotlib.pyplot as plt
import prediction


class History:
    def __init__(self, model_name, db, collection, batch_size, learning_rate):
        self.model_name = model_name
        self.db = db
        self.collection = collection
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.iter = []
        self.loss = []
        self.accuracy = []
        self.tuning = {}
        self.epoch_loss_gradients = {}
        self.epoch_accuracy_gradients = {}

    def batch_size_info(self):
        return '%s (%s%%)' \
               % (self.batch_size,
                  round(100 * self.batch_size
                        / stats.COLLECTION_SIZE[self.db][self.collection],
                        2))

    def compare(self, comparators, value_to_compare):
        # loss
        lines_loss = []
        plt.subplot(1, 2, 1)
        line_1_loss, = plt.plot(np.array(self.iter),
                                self.comparison_metric('loss'),
                                label=self.comparison_label(value_to_compare))
        lines_loss.append(line_1_loss)
        for i in range(len(comparators)):
            line_i_loss, = plt.plot(
                np.array(comparators[i].iter),
                comparators[i].comparison_metric('loss'),
                label=comparators[i].comparison_label(value_to_compare))
            lines_loss.append(line_i_loss)
        plt.legend(handles=lines_loss, loc=2)
        plt.xlabel('iteration')
        plt.ylabel('standardized loss')
        # accuracy
        lines_accuracy = []
        plt.subplot(1, 2, 2)
        line_1_accuracy, = plt.plot(
            np.array(self.iter),
            self.comparison_metric('accuracy'),
            label=self.comparison_label(value_to_compare))
        lines_accuracy.append(line_1_accuracy)
        for i in range(len(comparators)):
            line_i_accuracy, = plt.plot(
                np.array(comparators[i].iter),
                comparators[i].comparison_metric('accuracy'),
                label=comparators[i].comparison_label(value_to_compare))
            lines_accuracy.append(line_i_accuracy)
        plt.legend(handles=lines_accuracy, loc=2)
        plt.xlabel('iteration')
        plt.ylabel('accuracy')
        plt.show()

    def comparison_label(self, value_to_compare):
        if value_to_compare == 'batch_size':
            return self.batch_size_info()
        elif value_to_compare == 'learning_rate':
            return '%s' % self.learning_rate
        else:
            raise Exception('Unexpected value_to_compare: %s'
                            % value_to_compare)

    def comparison_metric(self, metric):
        if metric == 'loss':
            return self.standardized_loss()
        elif metric == 'accuracy':
            return np.array(self.accuracy)
        else:
            raise Exception('Unexpected metric: %s' % metric)

    def report(self, iter, loss, accuracy):
        self.iter.append(iter)
        self.loss.append(loss)
        self.accuracy.append(accuracy)

    def report_epoch_gradients(self, epoch, loss, accuracy):
        self.epoch_loss_gradients[epoch] = loss
        self.epoch_accuracy_gradients[epoch] = accuracy

    def report_tuning(self, iter, accuracy):
        self.tuning[iter] = accuracy

    def save(self):
        date_and_time = datetime.datetime.now().strftime('%d-%b-%Y_%H-%M-%S')
        util.save_pickle(self,
                         'histories/%s_%s_%s_%.2E_%s.pkl'
                         % (self.model_name,
                            self.collection,
                            self.batch_size,
                            self.learning_rate,
                            date_and_time))

    def standardized_loss(self):
        return np.log(np.array(self.loss) / self.batch_size)

    def visualize(self):
        plt.subplot(1, 2, 1)
        loss, = plt.plot(np.array(self.iter),
                         self.standardized_loss(),
                         label='loss')
        accuracy, = plt.plot(np.array(self.iter),
                             self.accuracy,
                             label='accuracy')
        tuning, = plt.plot(np.array(list(self.tuning.keys())),
                           np.array(list(self.tuning.values())),
                           label='tuning accuracy')
        plt.legend(handles=[loss, accuracy, tuning], loc=2)
        plt.subplot(1, 2, 2)
        epoch_loss, = plt.plot(
            np.array(list(self.epoch_loss_gradients.keys())),
            np.array(list(self.epoch_loss_gradients.values())),
            label='roc loss')
        epoch_accuracy, = plt.plot(
            np.array(list(self.epoch_accuracy_gradients.keys())),
            np.array(list(self.epoch_accuracy_gradients.values())),
            label='roc accuracy')
        plt.legend(handles=[epoch_loss, epoch_accuracy], loc=1)
        plt.show()


def report_every(num_iters):
    return np.floor(num_iters / 10)


def tune_every(num_epochs):
    return max(np.floor(num_epochs / 20), 1)


def train(model, db, collection, num_epochs, sess,
          batch_size=4, subset_size=None, tuning_collection=None,
          load_ckpt=True, save_ckpt=True, transfer=False):
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
    num_iters = batching.num_iters(db, collection, batch_size, subset_size)
    epoch_time_takens = []
    iter_time_takens = []
    history = History(model.name,
                      db,
                      collection,
                      batch_size,
                      model.config.learning_rate)
    epoch_starting_loss = 0.0
    epoch_starting_accuracy = 0.0
    epoch_final_loss = 0.0
    epoch_final_accuracy = 0.0
    epoch_change_loss = 0.0
    epoch_change_accuracy = 0.0
    for epoch in range(num_epochs):
        print('Epoch %s/%s\t\tloss\taccuracy\tavg(t)\tremaining'
              % (epoch + 1, num_epochs))
        epoch_start = time.time()
        batch_gen = batching.get_batch_gen(db,
                                           collection,
                                           batch_size=batch_size)
        last_iter = starting_iter + ((epoch + 1) * num_iters)
        while iter < last_iter:
            iter_start = time.time()
            batch = next(batch_gen)
            batch_loss, batch_accuracy, _ \
                = sess.run([model.loss,
                            model.accuracy,
                            model.optimize],
                           util.feed_dict(model, batch))
            if not start_reported:
                print('1\t\t%s\t%s%%'
                      % (batch_loss, round(batch_accuracy, 4)))
                epoch_starting_loss = batch_loss
                epoch_starting_accuracy = batch_accuracy
                start_reported = True
            average_loss += batch_loss
            average_accuracy += batch_accuracy
            history.report(iter, average_loss / iter, average_accuracy / iter)
            iter_end = time.time()
            iter_time_taken = iter_end - iter_start
            iter_time_takens.append(iter_time_taken)
            iter += 1
            average_time = np.average(iter_time_takens)
            iters_remaining = last_iter - iter
            if iter % report_every(num_iters) == 0:
                print('Step %s:\t'
                      '%s\t'
                      '%s%%\t'
                      '%ss\t'
                      '%ss' % (iter,
                               average_loss / iter,
                               round(average_accuracy / iter * 100, 4),
                               round(average_time, 2),
                               int(round(average_time * iters_remaining, 0))))
                if save_ckpt:
                    util.save_checkpoint(model,
                                         saver,
                                         sess,
                                         transfer,
                                         iter)
            if iter + 1 == last_iter:
                epoch_final_loss = average_loss / iter
                epoch_final_accuracy = average_accuracy / iter
                epoch_change_loss = \
                    epoch_final_loss - epoch_starting_loss
                epoch_change_accuracy = \
                    (epoch_final_accuracy - epoch_starting_accuracy) * 100
                epoch_starting_loss = average_loss / iter
                epoch_starting_accuracy = average_accuracy / iter
        epoch_end = time.time()
        epoch_time_taken = epoch_end - epoch_start
        epoch_time_takens.append(epoch_time_taken)
        history.report_epoch_gradients(epoch + 1,
                                       epoch_change_loss,
                                       epoch_change_accuracy)
        print('\t\t%s%s\t%s%s%%\t%ss\t%ss'
              % ('+' if epoch_change_loss > 0 else None,
                 epoch_change_loss,
                 '+' if epoch_change_accuracy > 0 else None,
                 round(epoch_change_accuracy, 2),
                 round(np.average(epoch_time_takens), 2),
                 round(np.average(epoch_time_takens
                                  * (num_epochs - epoch + 1)), 2)))
        if iter % tune_every(num_epochs) == 0:
            tuning_accuracy = prediction.accuracy(model=model,
                                                  db=db,
                                                  collection=tuning_collection,
                                                  sess=sess,
                                                  load_ckpt=False,
                                                  transfer=False)
            history.report_tuning(iter, tuning_accuracy)
    writer.close()
    model.in_training = False
    history.save()
    history.visualize()


if __name__ == '__main__':
    h = util.load_pickle(
        'histories/bi_rnn_alignment_dev_100_5.00E-04_28-Apr-2017_14-53-21.pkl')
    h.visualize()
