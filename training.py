import tensorflow as tf
from batching import get_batch_gen, NUM_ITERS, REPORT_EVERY, BATCH_SIZE
from util import feed_dict, feed_dict2, load_checkpoint, save_checkpoint, log_graph_path


def train(model, db, collection, num_epochs, sess, load_ckpt=True, save_ckpt=True, write_graph=True, transfer=False):
    # make sure sess.run(tf.global_variables_initializer() has already been run)
    if write_graph:
        writer = tf.summary.FileWriter(log_graph_path(model.name), sess.graph)
    saver = tf.train.Saver()
    if load_ckpt:
        load_checkpoint(model, saver, sess, transfer)
    model.in_training = True
    start_reported = False
    for epoch in range(num_epochs):
        print('Epoch %s' % (epoch + 1))
        batch_gen = get_batch_gen(db, collection)
        average_loss = 0
        average_accuracy = 0.0
        starting_point = model.global_step.eval()
        iteration = model.global_step.eval()
        while iteration < (starting_point + NUM_ITERS[db][collection]):
            batch = next(batch_gen)
            batch_loss, batch_accuracy, _ = sess.run([model.loss, model.accuracy, model.optimize],
                                                        feed_dict(model, batch))
            if not start_reported:
                print('Starting condition: loss = %s; accuracy = %s' % (batch_loss, batch_accuracy))
                start_reported = True
            average_loss += batch_loss
            average_accuracy += batch_accuracy
            if (iteration + 1) % REPORT_EVERY[db][collection] == 0:
                print('Step %s: average loss = %s; average accuracy = %s' % (iteration + 1,
                                                                             average_loss / (iteration + 1),
                                                                             average_accuracy / (iteration + 1)))
                if save_ckpt:
                    save_checkpoint(model, saver, sess, transfer)
            iteration += 1
    if write_graph:
        writer.close()
    model.in_training = False


def train2(model, batch_gen_wrapper, num_epochs, sess,
           load_ckpt=True, save_ckpt=True, write_graph=True, transfer=False):
    # make sure sess.run(tf.global_variables_initializer() has already been run)
    loss_history = []
    accuracy_train_history = []
    if write_graph:
        writer = tf.summary.FileWriter(log_graph_path(model.name), sess.graph)
    saver = tf.train.Saver()
    if load_ckpt:
        load_checkpoint(model, saver, sess, transfer)
    for epoch in range(num_epochs):
        print('Epoch %s' % (epoch + 1))
        batch_gen = batch_gen_wrapper.new_batch_generator()
        cumulative_loss = 0
        cumulative_accuracy = 0.0
        starting_point = model.global_step.eval()
        iteration = model.global_step.eval()
        while iteration < (starting_point + batch_gen_wrapper.num_iters):
            batch = next(batch_gen)
            batch_loss, batch_accuracy, _ = sess.run([model.loss, model.accuracy_train, model.optimize],
                                                     feed_dict2(model, batch))
            cumulative_loss += batch_loss
            cumulative_accuracy += batch_accuracy
            average_loss = cumulative_loss / (iteration + 1)
            average_accuracy = cumulative_accuracy / (iteration + 1)
            loss_history.append(average_loss)
            accuracy_train_history.append(average_accuracy)
            if (iteration + 1) % batch_gen_wrapper.report_every == 0:
                print('Step %s: average loss = %s; average accuracy = %s' % (iteration + 1,
                                                                             average_loss,
                                                                             average_accuracy))
                if save_ckpt:
                    save_checkpoint(model, saver, sess, iteration, transfer)
            iteration += 1
    if write_graph:
        writer.close()
    return loss_history, accuracy_train_history
