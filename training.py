import tensorflow as tf
from process_data import get_batch_gen, NUM_ITERS, REPORT_EVERY
from util import feed_dict, load_checkpoint, save_checkpoint


def train(model, collection, num_epochs, load_ckpt=True, transfer=False):
    #sess_config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        load_checkpoint(model, saver, sess, load_ckpt, transfer)
        for epoch in range(num_epochs):
            print('Epoch %s' % (epoch + 1))
            batch_gen = get_batch_gen(collection)
            average_loss = 0
            average_accuracy = 0.0
            starting_point = model.global_step.eval()
            iteration = model.global_step.eval()
            while iteration < (starting_point + NUM_ITERS[collection]):
                batch = next(batch_gen)
                batch_loss, batch_accuracy, _ = sess.run([model.loss, model.accuracy_train, model.optimize],
                                                         feed_dict(model, batch))
                average_loss += batch_loss
                average_accuracy += batch_accuracy
                if (iteration + 1) % REPORT_EVERY[collection] == 0:
                    print('Step %s: average loss = %s; average accuracy = %s' % (iteration + 1,
                                                                                 average_loss / (iteration + 1),
                                                                                 average_accuracy / (iteration + 1)))
                    save_checkpoint(model, saver, sess, iteration, transfer)
                iteration += 1
