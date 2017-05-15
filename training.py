import tensorflow as tf
import numpy as np
import batching
import util
import time
import prediction
import hist


DEBUG = False


def report_every(num_iters):
    return np.floor(num_iters / 10)


def tune_every(num_epochs):
    return int(min(max(np.floor(num_epochs / 20), 1), 10))


def train(model, db, collection, num_epochs, sess,
          batch_size=4, subset_size=None, tuning_collection=None,
          load_ckpt=True, save_ckpt=True, transfer=False):

    print('------\n'
          'Training %s on %s%s.%s for %s epochs '
          'with batch size %s at LR %s'
          % (model.name,
             ('' if not subset_size else '%s samples of ' % subset_size),
             db,
             collection,
             num_epochs,
             batch_size,
             model.learning_rate))

    # num_epochs is a target - we keep the state in global_epoch

    # NOTE: make sure sess.run(tf.global_variables_initializer())
    #       has already been run

    # self-explanatory variables we need for the process
    writer = tf.summary.FileWriter(util.log_graph_path(model.name), sess.graph)
    saver = tf.train.Saver(max_to_keep=10000)
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
    tuning_iter = model.tuning_iter.eval()
    accumulated_tuning_accuracy = model.accumulated_tuning_accuracy.eval()
    training_history_id = model.training_history_id.eval()

    # variables required for tracking tuning set accuracy
    previous_training_accuacy = \
        (accumulated_tuning_accuracy / tuning_iter) \
            if tuning_iter > 0 \
            else 0.0
    average_tuning_accuracy = 0.0
    change_in_tuning_accuracy = 0.0

    # if we don't have a training history id, create new and get the id
    if training_history_id < 0:  # init to -1 in constructor for no history
        training_history_id = hist.new_history(
            model.name,
            db,
            collection,
            batch_size,
            subset_size,
            model.config)
        sess.run(
            model.set_training_history_id,
            {model.new_training_history_id: training_history_id})

    # summary variables for History
    epoch_time_takens = []
    iter_time_takens = []  # averaging these over the whole process
                           # not just epochs

    # START EPOCHS
    while epoch < num_epochs:
        # we init to zero, and update after each, so this is correct
        epoch += 1

        # print the header with dividers for visual ease
        print_dividing_lines()
        print('Epoch %s/%s\tloss\t\taccuracy\tavg(t)\t\tremaining'
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
        batch_gen = batching.get_batch_gen(db=db,
                                           collection=collection,
                                           batch_size=batch_size)

        # START ITERS
        while iter < epoch_last_iter:
            # global_step is initialized to zero - iterating here is correct
            iter += 1

            # take the time before starting
            iter_start = time.time()

            # get the next batch and run the ops
            # NOTE: taking a feed_dict function as an argument
            #       could genericize this
            batch = next(batch_gen)

            # get the right optimization op
            optimize_op = model.optimize
            if transfer:
                optimize_op = model.optimize_transfer

            batch_loss, batch_accuracy, _ \
                = sess.run([model.loss,
                            model.accuracy,
                            optimize_op],
                           util.feed_dict(model, batch))

            # accumulate the loss and accuracy
            accumulated_loss += batch_loss
            accumulated_accuracy += batch_accuracy

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
                hist.report_batch(
                    training_history_id,
                    iter,
                    accumulated_loss / iter,
                    accumulated_accuracy / iter)
                print('Step %s:\t'
                      '%8.5f\t'
                      '%6.4f%%\t'
                      '%s\t'
                      '%s' % (iter,
                              accumulated_loss / iter,
                              accumulated_accuracy / iter * 100,
                              pretty_time(average_time),
                              pretty_time(average_time * iters_remaining)))

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
            sess.run([model.update_iter,
                      model.update_loss,
                      model.update_accuracy],
                     {model.new_global_step: iter,
                      model.new_accumulated_loss: accumulated_loss,
                      model.new_accumulated_accuracy: accumulated_accuracy})

            # END ITER
        # END ITERS

        # save the global_epoch state to the model
        sess.run(model.update_epoch,
                 {model.new_global_epoch: epoch})

        # calculate epoch stats
        epoch_end = time.time()
        epoch_time_taken = epoch_end - epoch_start
        epoch_time_takens.append(epoch_time_taken)
        hist.report_epoch(
            training_history_id,
            epoch,
            epoch_change_average_loss,
            epoch_change_average_accuracy)

        # print the results
        print_dividing_lines()
        print('\t\t%s%10.5f\t%s%6.4f%%\t%s\t%s'
              % ('+' if epoch_change_average_loss > 0 else '',
                 epoch_change_average_loss,
                 '+' if epoch_change_average_accuracy > 0 else '',
                 epoch_change_average_accuracy,
                 pretty_time(np.average(epoch_time_takens)),
                 pretty_time(np.average(epoch_time_takens)
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
            tuning_iter += 1
            tuning_accuracy = prediction.accuracy(model=model,
                                                  db=db,
                                                  collection=tuning_collection,
                                                  sess=sess,
                                                  surpress_print=True)
            accumulated_tuning_accuracy += tuning_accuracy
            sess.run(
                [model.update_tuning_iter,
                 model.update_tuning_accuracy],
                {model.new_tuning_iter: tuning_iter,
                 model.new_accumulated_tuning_accuracy: accumulated_tuning_accuracy})
            average_tuning_accuracy = accumulated_tuning_accuracy / tuning_iter
            hist.report_tuning(training_history_id,
                               tuning_iter,
                               average_tuning_accuracy)
            change_in_tuning_accuracy = \
                average_tuning_accuracy - previous_training_accuacy
            previous_training_accuacy = average_tuning_accuracy
            print('Average tuning accuracy: %5.3f%% (%s%5.3f%%)' %
                  (average_tuning_accuracy * 100,
                   '+' if change_in_tuning_accuracy > 0 else '',
                   change_in_tuning_accuracy * 100))

        # END EPOCH
    # END EPOCHS

    # clean up writer object
    writer.close()


def print_dividing_lines():
    print('------\t\t------\t\t------\t\t------\t\t------')


def report(info):
    if DEBUG:
        print(info)


def pretty_time(secs):
    if secs < 60.0:
        return '%4.2f secs' % secs
    elif secs < 3600.0:
        return '%4.2f mins' % (secs / 60)
    elif secs < 86400.0:
        return '%4.2f hrs' % (secs / 60 / 60)
    else:
        return '%3.2f days' % (secs / 60 / 60 / 24)
