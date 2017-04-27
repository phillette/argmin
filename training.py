import tensorflow as tf
import numpy as np
import batching
import util
import stats
import time
import datetime
import matplotlib.pyplot as plt


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

    def batch_size_info(self):
        return '%s (%s%%)' % (self.batch_size,
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
            line_i_loss, = plt.plot(np.array(comparators[i].iter),
                                    comparators[i].comparison_metric('loss'),
                                    label=comparators[i].comparison_label(value_to_compare))
            lines_loss.append(line_i_loss)
        plt.legend(handles=lines_loss, loc=2)
        plt.xlabel('iteration')
        plt.ylabel('standardized loss')
        # accuracy
        lines_accuracy = []
        plt.subplot(1, 2, 2)
        line_1_accuracy, = plt.plot(np.array(self.iter),
                                    self.comparison_metric('accuracy'),
                                    label=self.comparison_label(value_to_compare))
        lines_accuracy.append(line_1_accuracy)
        for i in range(len(comparators)):
            line_i_accuracy, = plt.plot(np.array(comparators[i].iter),
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
            raise Exception('Unexpected value_to_compare: %s' % value_to_compare)

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

    def save(self):
        date_and_time = datetime.datetime.now().strftime('%d-%b-%Y_%H-%M-%S')
        util.save_pickle(self, 'histories/%s_%s_%s_%.2E_%s.pkl' % (self.model_name,
                                                                   self.collection,
                                                                   self.batch_size,
                                                                   self.learning_rate,
                                                                   date_and_time))

    def standardized_loss(self):
        return np.log(np.array(self.loss) / self.batch_size)

    def visualize(self):
        loss, = plt.plot(np.array(self.iter), self.standardized_loss(), label='loss')
        accuracy, = plt.plot(np.array(self.iter), self.accuracy, label='accuracy')
        plt.legend(handles=[loss, accuracy], loc=2)
        plt.show()


def report_every(num_iters):
    return np.floor(num_iters / 10)


def train(model, db, collection, num_epochs, sess, batch_size=4,
          load_ckpt=True, save_ckpt=True, transfer=False, summarise=False):
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
    history = History(model.name, db, collection, batch_size, model.config.learning_rate)
    for epoch in range(num_epochs):
        print('Epoch %s/%s\t\tloss\taccuracy\tavg(t)\tremaining' % (epoch + 1, num_epochs))
        epoch_start = time.time()
        batch_gen = batching.get_batch_gen(db, collection, batch_size=batch_size)
        last_iter = starting_iter + ((epoch + 1) * num_iters)
        while iter < last_iter:
            iter_start = time.time()
            batch = next(batch_gen)
            batch_loss, batch_accuracy, _ = sess.run([model.loss,
                                                      model.accuracy,
                                                      model.optimize],
                                                     util.feed_dict(model, batch))
            if not start_reported:
                print('Starting condition: loss = %s; accuracy = %s' % (batch_loss, batch_accuracy))
                start_reported = True
            average_loss += batch_loss
            average_accuracy += batch_accuracy
            history.report(iter, average_loss / iter, average_accuracy / iter)
            iter_end = time.time()
            iter_time_taken = iter_end - iter_start
            iter_time_takens.append(iter_time_taken)
            #if summarise:
            #    writer.add_summary(summary, global_step=model.global_step.eval())
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
                    util.save_checkpoint(model, saver, sess, transfer, iter)
        epoch_end = time.time()
        epoch_time_taken = epoch_end - epoch_start
        epoch_time_takens.append(epoch_time_taken)
        print('Epoch time: %s; average = %s' % (round(epoch_time_taken, 2),
                                                round(np.average(epoch_time_takens), 2)))
    writer.close()
    model.in_training = False
    history.visualize()
    history.save()


if __name__ == '__main__':
    a = util.load_pickle('histories/alignment_dev_4_1.00E-03_27-Apr-2017_14-20-50.pkl')
    b = util.load_pickle('histories/alignment_dev_8_1.00E-03_27-Apr-2017_14-31-31.pkl')
    c = util.load_pickle('histories/alignment_dev_16_1.00E-03_27-Apr-2017_15-04-26.pkl')
    d = util.load_pickle('histories/alignment_dev_32_1.00E-03_27-Apr-2017_10-54-06.pkl')
    e = util.load_pickle('histories/alignment_dev_64_1.00E-03_27-Apr-2017_14-45-41.pkl')
    f = util.load_pickle('histories/alignment_dev_128_1.00E-03_27-Apr-2017_14-56-25.pkl')
    g = util.load_pickle('histories/alignment_dev_256_1.00E-03_27-Apr-2017_15-06-00.pkl')
    h = util.load_pickle('histories/alignment_dev_512_1.00E-03_27-Apr-2017_16-08-46.pkl')
    i = util.load_pickle('histories/alignment_dev_1024_1.00E-03_27-Apr-2017_15-23-01.pkl')
    a.compare([b, c, d, e, f, g, h, i], 'batch_size')
