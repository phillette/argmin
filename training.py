import tensorflow as tf
from batching import get_batch_gen
from stats import *
from util import feed_dict, load_checkpoint, save_checkpoint, log_graph_path
import objgraph


# def train(model, db, collection, num_epochs, sess, load_ckpt=True, save_ckpt=True, transfer=False, summarise=False):
#     # make sure sess.run(tf.global_variables_initializer() has already been run)
#     writer = tf.summary.FileWriter(log_graph_path(model.name), sess.graph)
#     saver = tf.train.Saver()
#     if load_ckpt:
#         load_checkpoint(model, saver, sess, transfer)
#     model.in_training = True
#     start_reported = False
#     average_loss = 0.0
#     average_accuracy = 0.0
#     for epoch in range(num_epochs):
#         global_step_starting_iter = model.global_step.eval()
#         print('Epoch %s' % (epoch + 1))
#         batch_gen = get_batch_gen(db, collection)
#         while model.global_step.eval() < global_step_starting_iter + NUM_ITERS[db][collection]:
#             batch = next(batch_gen)
#             batch_loss, batch_accuracy, _ = sess.run([model.loss,
#                                                       model.accuracy,
#                                                       model.optimize],
#                                                      feed_dict(model, batch))
#             if not start_reported:
#                 print('Starting condition: loss = %s; accuracy = %s' % (batch_loss, batch_accuracy))
#                 start_reported = True
#             average_loss += batch_loss
#             average_accuracy += batch_accuracy
#             #if summarise:
#                 #writer.add_summary(summary, global_step=model.global_step.eval())
#             model.global_step += 1
#             if (model.global_step.eval()) % REPORT_EVERY[db][collection] == 0:
#                 print('Step %s: loss = %s; accuracy = %s' % (model.global_step.eval(),
#                                                              average_loss / (model.global_step.eval()),
#                                                              average_accuracy / (model.global_step.eval())))
#                 if save_ckpt:
#                     save_checkpoint(model, saver, sess, transfer)
#             print('************************')
#             objgraph.show_most_common_types(limit=20)
#     writer.close()
#     model.in_training = False


def train(model, db, collection, num_epochs, sess, load_ckpt=True, save_ckpt=True, transfer=False, summarise=False):
    # make sure sess.run(tf.global_variables_initializer() has already been run)
    writer = tf.summary.FileWriter(log_graph_path(model.name), sess.graph)
    saver = tf.train.Saver()
    if load_ckpt:
        load_checkpoint(model, saver, sess, transfer)
    model.in_training = True
    start_reported = False
    average_loss = 0.0
    average_accuracy = 0.0
    for epoch in range(num_epochs):
        iter = 0
        global_step_starting_iter = 0
        print('Epoch %s' % (epoch + 1))
        batch_gen = get_batch_gen(db, collection)
        while iter < global_step_starting_iter + NUM_ITERS[db][collection]:
            batch = next(batch_gen)
            batch_loss, batch_accuracy, _ = sess.run([model.loss,
                                                      model.accuracy,
                                                      model.optimize],
                                                     feed_dict(model, batch))
            if not start_reported:
                print('Starting condition: loss = %s; accuracy = %s' % (batch_loss, batch_accuracy))
                start_reported = True
            average_loss += batch_loss
            average_accuracy += batch_accuracy
            #if summarise:
                #writer.add_summary(summary, global_step=model.global_step.eval())
            iter += 1
            if iter % REPORT_EVERY[db][collection] == 0:
                print('Step %s: loss = %s; accuracy = %s' % (iter,
                                                             average_loss / iter,
                                                             average_accuracy / iter))
                if save_ckpt:
                    save_checkpoint(model, saver, sess, transfer)
            print('************************')
            objgraph.show_most_common_types(limit=20)
    writer.close()
    model.in_training = False
