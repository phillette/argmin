import tensorflow as tf
import numpy as np
import batching
import util
import stats
import time
import datetime
import matplotlib.pyplot as plt
import prediction


DEBUG = False


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
        elif value_to_compare == 'model_name':
            return '%s' % self.model_name
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
        plt.legend(handles=[loss, accuracy, tuning], loc=1)
        plt.subplot(1, 2, 2)
        epoch_loss, = plt.plot(
            np.array(list(self.epoch_loss_gradients.keys())),
            np.array(list(self.epoch_loss_gradients.values())),
            label='roc loss')
        epoch_accuracy, = plt.plot(
            np.array(list(self.epoch_accuracy_gradients.keys())),
            np.array(list(self.epoch_accuracy_gradients.values())),
            label='roc accuracy')
        plt.legend(handles=[epoch_loss, epoch_accuracy], loc=2)
        plt.show()


def report_every(num_iters):
    return np.floor(num_iters / 10)


def tune_every(num_epochs):
    return int(min(max(np.floor(num_epochs / 20), 1), 10))


def train(model, db, collection, num_epochs, sess,
          batch_size=4, subset_size=None, tuning_collection=None,
          load_ckpt=True, save_ckpt=True, transfer=False):

    print('Training %s on %s.%s for %s epochs '
          'with batch size %s at LR %s'
          % (model.name, db, collection,
             num_epochs, batch_size, model.config.learning_rate))

    # num_epochs is a target - we keep the state in global_epoch

    # NOTE: make sure sess.run(tf.global_variables_initializer())
    #       has already been run

    # self-explanatory variables we need for the process
    writer = tf.summary.FileWriter(util.log_graph_path(model.name), sess.graph)
    saver = tf.train.Saver(max_to_keep=10000)
    history = History(model.name,
                      db,
                      collection,
                      batch_size,
                      model.config.learning_rate)
    num_iters = batching.num_iters(db=db,
                                   collection=collection,
                                   batch_size=batch_size,
                                   subset_size=subset_size)

    # load the model checkpoint if required
    if load_ckpt:
        util.load_checkpoint(model, saver, sess, transfer)

    # initialize training variables according to model state
    epoch = model.global_epoch.eval()
    iter = model.global_step.eval()
    accumulated_loss = model.accumulated_loss.eval()
    accumulated_accuracy = model.accumulated_accuracy.eval()

    # define the update ops for the training state variables
    ph_global_epoch = tf.placeholder(tf.int32)
    ph_global_step = tf.placeholder(tf.int32)
    ph_accumulated_loss = tf.placeholder(tf.float32)
    ph_accumulated_accuracy = tf.placeholder(tf.float32)
    update_epoch = tf.assign(model.global_epoch,
                             ph_global_epoch)
    update_iter = tf.assign(model.global_step,
                            ph_global_step)
    update_loss = tf.assign(model.accumulated_loss,
                            ph_accumulated_loss)
    update_accuracy = tf.assign(model.accumulated_accuracy,
                                ph_accumulated_accuracy)

    # summary variables for History
    epoch_time_takens = []
    iter_time_takens = []  # averaging these over the whole process
                           # not just epochs

    report('Global initialization successful. '
           'Model state: epoch = %s; iter = %s'
           % (epoch, iter))

    # START EPOCHS
    while epoch < num_epochs:
        # we init to zero, and update after each, so this is correct
        epoch += 1

        # print the header with dividers for visual ease
        print_dividing_lines()
        print('Epoch %s/%s\tloss\t\taccuracy\tavg(t)\tremaining'
              % (epoch, num_epochs))
        print_dividing_lines()

        # variables to hold epoch statistics
        epoch_start = time.time()
        epoch_last_iter = iter + num_iters
        epoch_start_average_loss = 0.0
        epoch_start_average_accuracy = 0.0
        epoch_end_average_loss = 0.0
        epoch_end_average_accuracy = 0.0
        epoch_change_average_loss = 0.0
        epoch_change_average_accuracy = 0.0
        first_report_made = False

        # get the batch generator for this epoch
        # NOTE: a generator generator could genericize this,
        #       untying it from this structure
        report('Getting batch_gen...')
        batch_gen = batching.get_batch_gen(db,
                                           collection,
                                           batch_size=batch_size)
        report('Successfully got batch_gen.')

        # START ITERS
        while iter < epoch_last_iter:
            # global_step is initialized to zero - iterating here is correct
            iter += 1
            report('iter: %s' % iter)

            # take the time before starting
            iter_start = time.time()

            # get the next batch and run the ops
            # NOTE: taking a feed_dict function as an argument
            #       could genericize this
            report('Getting next batch...')
            batch = next(batch_gen)
            report('Successfully got batch. Printing details.')
            report(batch.details())
            report('Running ops...')
            batch_loss, batch_accuracy, _ \
                = sess.run([model.loss,
                            model.accuracy,
                            model.optimize],
                           util.feed_dict(model, batch))
            report('Ops ran successfully.')

            # accumulate the loss and accuracy
            accumulated_loss += batch_loss
            accumulated_accuracy += batch_accuracy

            # report information to History
            # NOTE: I'm wondering if this could be tied to the model?
            history.report(iter,
                           accumulated_loss / iter,
                           accumulated_accuracy / iter)

            # calculate time related variables and report
            iter_end = time.time()
            iter_time_taken = iter_end - iter_start
            iter_time_takens.append(iter_time_taken)
            average_time = np.average(iter_time_takens)
            iters_remaining = epoch_last_iter - iter

            # print to screen if it's time
            if iter % report_every(num_iters) == 0:
                if not first_report_made:
                    epoch_start_average_loss = accumulated_loss / iter
                    epoch_start_average_accuracy = accumulated_accuracy / iter
                    first_report_made = True
                print('Step %s:\t'
                      '%10.5f\t'
                      '%6.4f%%\t'
                      '%6.2f\t'
                      '%s' % (iter,
                              accumulated_loss / iter,
                              accumulated_accuracy / iter * 100,
                              average_time,
                              time_remaining(average_time * iters_remaining)))

            # if we're in the last iteration, update end of epoch stats
            if iter == epoch_last_iter:
                epoch_end_average_loss = accumulated_loss / iter
                epoch_end_average_accuracy = accumulated_accuracy / iter
                epoch_change_average_loss = \
                    epoch_end_average_loss - epoch_start_average_loss
                epoch_change_average_accuracy = \
                    (epoch_end_average_accuracy
                     - epoch_start_average_accuracy) \
                    * 100

            # update the training state variables on the model
            sess.run([update_iter, update_loss, update_accuracy],
                     {ph_global_step: iter,
                      ph_accumulated_loss: accumulated_loss,
                      ph_accumulated_accuracy: accumulated_accuracy})

            # END ITER
        # END ITERS

        # save the global_epoch state to the model
        sess.run(update_epoch, {ph_global_epoch: epoch})

        # calculate epoch stats
        epoch_end = time.time()
        epoch_time_taken = epoch_end - epoch_start
        epoch_time_takens.append(epoch_time_taken)
        history.report_epoch_gradients(epoch,
                                       epoch_change_average_loss,
                                       epoch_change_average_accuracy)

        # print the results
        print_dividing_lines()
        print('\t\t%s%10.5f\t%s%6.4f%%\t%8.2fs\t%s'
              % ('+' if epoch_change_average_loss > 0 else '',
                 epoch_change_average_loss,
                 '+' if epoch_change_average_accuracy > 0 else '',
                 epoch_change_average_accuracy,
                 np.average(epoch_time_takens),
                 time_remaining(np.average(epoch_time_takens)
                                           * (num_epochs - epoch))))

        # save the checkpoint if required
        if save_ckpt:
            util.save_checkpoint(model=model,
                                 saver=saver,
                                 sess=sess,
                                 global_step=iter,
                                 transfer=transfer)

        # perform tuning on dev set if required and if its time
        if tuning_collection and iter % tune_every(num_epochs) == 0:
            tuning_accuracy = prediction.accuracy(model=model,
                                                  db=db,
                                                  collection=tuning_collection,
                                                  sess=sess,
                                                  load_ckpt=False,
                                                  transfer=False)
            history.report_tuning(iter, tuning_accuracy)

        # END EPOCH
    # END EPOCHS

    # clean up writer object
    writer.close()

    # save and report history
    history.save()
    history.visualize()


def print_dividing_lines():
    print('------\t\t------\t\t------\t\t------\t------')


def report(info):
    if DEBUG:
        print(info)


def time_remaining(secs):
    if secs < 60.0:
        return '%4.2f secs' % secs
    elif secs < 3600.0:
        return '%4.2f mins' % (secs / 60)
    elif secs < 86400.0:
        return '%4.2f hrs' % (secs / 60 / 60)
    else:
        return '%3.2f days' % (secs / 60 / 60 / 24)


if __name__ == '__main__':
    h = util.load_pickle(
        'histories/bi_rnn_alignment_dev_100_5.00E-04_28-Apr-2017_14-53-21.pkl')
    h.visualize()
