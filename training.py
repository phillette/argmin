import tensorflow as tf
import os


def train(model, batch_gen, learning_rate=0.01, num_iters=1000, report_every=100, load_ckpt=True):
    #sess_config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session() as sess:
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(model.loss)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/%s/%s.ckpt' % (model.name, model.name)))
        if load_ckpt and ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        average_loss = 0
        average_accuracy = 0.0
        starting_point = model.global_step.eval()
        iteration = model.global_step.eval()
        while iteration < (starting_point + num_iters):
            batch = next(batch_gen)
            batch_loss, batch_accuracy, _ = sess.run([model.loss, model.accuracy, optimizer],
                                                     {
                                                         model.premises: batch.premises,
                                                         model.hypotheses: batch.hypotheses,
                                                         model.y: batch.labels
                                                     })
            average_loss += batch_loss
            average_accuracy += batch_accuracy
            if (iteration + 1) % report_every == 0:
                print('Step %s: average loss = %s; average accuracy = %s' % (iteration + 1,
                                                                             average_loss / (iteration + 1),
                                                                             average_accuracy / (iteration + 1)))
                saver.save(sess, 'checkpoints/%s/%s.ckpt' % (model.name, model.name), iteration)
            iteration += 1
