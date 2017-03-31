import tensorflow as tf
from batching import get_batch_gen, NUM_ITERS, REPORT_EVERY
from util import feed_dict, load_checkpoint, save_checkpoint, log_graph_path


def train(model, db, collection, num_epochs, sess, load_ckpt=True, save_ckpt=True, write_graph=True, transfer=False):
    # make sure sess.run(tf.global_variables_initializer() has already been run)
    if write_graph:
        writer = tf.summary.FileWriter(log_graph_path(model.name), sess.graph)
    saver = tf.train.Saver()
    if load_ckpt:
        load_checkpoint(model, saver, sess, transfer)
    for epoch in range(num_epochs):
        print('Epoch %s' % (epoch + 1))
        batch_gen = get_batch_gen(db, collection)
        average_loss = 0
        average_accuracy = 0.0
        starting_point = model.global_step.eval()
        iteration = model.global_step.eval()
        while iteration < (starting_point + NUM_ITERS[db][collection]):
            batch = next(batch_gen)
            batch_loss, batch_accuracy, _ = sess.run([model.loss, model.accuracy_train, model.optimize],
                                                     feed_dict(model, batch))
            average_loss += batch_loss
            average_accuracy += batch_accuracy
            if (iteration + 1) % REPORT_EVERY[db][collection] == 0:
                print('Step %s: average loss = %s; average accuracy = %s' % (iteration + 1,
                                                                             average_loss / (iteration + 1),
                                                                             average_accuracy / (iteration + 1)))
                if save_ckpt:
                    save_checkpoint(model, saver, sess, iteration, transfer)
            iteration += 1
    if write_graph:
        writer.close()
