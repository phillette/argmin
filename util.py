import tensorflow as tf
import os


def add_bias(tensor, dtype=tf.float64, axis=1):
    return tf.concat([tf.ones(dtype=dtype,
                              shape=[tf.shape(tensor)[0], 1]),
                      tensor],
                     axis=axis)


def ckpt_path(model_name, transfer=False):
    if transfer:
        return os.path.dirname('checkpoints/%s/transfer/%s' % (model_name, model_name))
    else:
        return os.path.dirname('checkpoints/%s/' % model_name)


def clip_gradients(grads_and_vars, norm=3.0, axes=0):
    return [(tf.clip_by_norm(gv[0], clip_norm=norm, axes=axes), gv[1]) for gv in grads_and_vars]


def dropout_vector(keep_prob, shape):
    return tf.where(condition=tf.random_uniform(shape, 0.0, 1.0, tf.float64) > 1 - keep_prob,
                    x=tf.ones(shape, tf.float64),
                    y=tf.zeros(shape, tf.float64))


def feed_dict(model, batch):
    return {model.X.premises: batch.premises,
            model.X.hypotheses: batch.hypotheses,
            model.Y: batch.labels}


def feed_dict2(model, batch):
    return {model.X: batch.X,
            model.Y: batch.Y}


def length(sequence):
    """
    Courtesy of Danijar Hafner:
    https://danijar.com/variable-sequence-lengths-in-tensorflow/
    """
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length


def load_checkpoint(model, saver, sess, transfer=False):
    ckpt = tf.train.get_checkpoint_state(ckpt_path(model.name, transfer))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path + '.ckpt')
    else:
        raise Exception('Checkpoint not found, not loaded.')


def log_graph_path(model_name):
    return os.path.dirname('graphs/%s' % model_name)


def save_checkpoint(model, saver, sess, iteration, transfer=False):
    path = ckpt_path(model.name, transfer) + '/%s' % model.name
    saver.save(sess, path, iteration)
