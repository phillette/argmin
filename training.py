import tensorflow as tf
import os


def train(model, batch_gen, num_iters=5000, report_every=500, load_ckpt=True):
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(sess_config) as sess:
        with tf.device('/gpu:0'):
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/%s.ckpt') % model.name)
            if load_ckpt and ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            average_loss = 0
            iteration = model.global_step.eval()
            while iteration < num_iters:
                batch = next(batch_gen)
                batch_loss, _ = sess.run([model.loss, model.optimize],
                                         {
                                             model.premises: batch.premises,
                                             model.hypotheses: batch.hypotheses,
                                             model.y: batch.y
                                         })
                average_loss += batch_loss
                if (iteration + 1) % report_every == 0:
                    print('Average loss at step %s: %s' % (iteration + 1,
                                                           average_loss / (iteration + 1)))
                    saver.save(sess, 'checkpoints/%s.ckpt' % model.name, iteration)
                iteration += 1
