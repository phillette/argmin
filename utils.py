import tensorflow as tf
import os
import pickle
import functools
import errors


PICKLE_DIR = 'pickles/'


def add_bias(tensor, dtype=tf.float64, axis=1):
    return tf.concat([tf.ones(dtype=dtype,
                              shape=[tf.shape(tensor)[0], 1]),
                      tensor],
                     axis=axis)


def ckpt_path(model_name, transfer=False):
    if transfer:
        return 'checkpoints/%s/transfer/%s' % (model_name, model_name)
    else:
        return 'checkpoints/%s/%s' % (model_name, model_name)


def clip_gradients(grads_and_vars, norm=3.0, axes=0):
    return [(tf.clip_by_norm(gv[0],
                             clip_norm=norm,
                             axes=axes), gv[1])
            for gv in grads_and_vars
            if gv[0] is not None]


def concat(premises, hypotheses):
    """
    Sometimes I need to put both premises and hypotheses tensors
    through the same fully connected layer. We can concatenate
    them along the batch dimension to achieve this.  This also
    applies to other tensors, alphas and betas in the alignment
    model, that are downstream modified tensors relating to
    premises and hypotheses.
    :param premises: tensor in [batch_size, a, b], where a is usually
                     going to be num_time_steps, and b embed_size
    :param hypotheses: tensor in [batch_size, a, b], where a is usually
                       going to be num_time_steps, and b embed_size
    :return: tensor of shape [2 * batch_size, a, b]
    """
    return tf.concat([premises, hypotheses], axis=0)


def dropout_vector(keep_prob, shape):
    return tf.where(condition=tf.random_uniform(
        shape, 0.0, 1.0, tf.float64) > 1 - keep_prob,
                    x=tf.ones(shape, tf.float64),
                    y=tf.zeros(shape, tf.float64))


def factors(n):
    return set(functools.reduce(list.__add__,
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))


def feed_dict(model, batch, in_training):
    training_flag = 1. if in_training else 0.
    return {model.premises: batch.premises,
            model.hypotheses: batch.hypotheses,
            model.Y: batch.labels,
            model.training_flag: training_flag}


def feed_dict_transfer(model, batch, training_flag):
    """Feed dict for transfer learning models.

    Args:
      model: a transfer learning model.
      batch: a transfer learning batch.
      training_flag: float equal to 1 if in training otherwise 0.
    """
    return {model.X: batch.X,
            model.Y: batch.labels,
            model.training_flag: training_flag}


def length(sequence):
    """
    Courtesy of Danijar Hafner:
    https://danijar.com/variable-sequence-lengths-in-tensorflow/
    """
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length


def load_checkpoint(model, saver, sess, transfer=False, path=None):
    if not path:
        path = ckpt_path(model.name, transfer)
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(path))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise Exception('Checkpoint "%s" not found' % path)


def load_checkpoint_at_step(model_name, global_step, saver, sess, path=None):
    if not path:
        path = ckpt_path(model_name, False)
    path += '-%s' % global_step
    saver.restore(sess, path)


def load_pickle(file_name):
    try:
        with open(PICKLE_DIR + file_name, 'rb') as file:
            obj = pickle.load(file)
            return obj
    except FileNotFoundError:
        raise errors.PickleNotFoundError(file_name)


def log_graph_path(model_name):
    return os.path.dirname('graphs/%s/%s' % (model_name, model_name))


def optimistic_restore(sess, save_file):
    """
    This not working currently.
    Investigation showed that saved_shapes is empty.
    It may have to do with the nature of the files I save.
    There may be a new way to save checkpoints that can actually
    work with this.
    """
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    print('saved_shapes')
    print(saved_shapes)
    var_names = sorted([(var.name, var.name.split(':')[0])
                        for var in tf.global_variables()
                        if var.name.split(':')[0] in saved_shapes])
    print(var_names)
    restore_vars = []
    name2var = dict(zip(map(lambda x: x.name.split(':')[0],
                            tf.global_variables()),
                        tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            current_var = name2var[saved_var_name]
            var_shape = current_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(current_var)
    saver = tf.train.Saver(restore_vars)
    saver.restore(sess, save_file)


def roll_batch(x, old_dims):
    return tf.reshape(x, old_dims)


def save_checkpoint(model, saver, sess, global_step, transfer=False):
    path = ckpt_path(model.name, transfer)
    saver.save(sess, path, global_step=global_step)


def save_pickle(obj, file_name):
    with open(PICKLE_DIR + file_name, 'wb') as file:
        pickle.dump(obj, file)


def scale_epoch_to_iter(epoch, iters_per_epoch):
    return epoch * iters_per_epoch


def scale_iter_to_epoch(iters, iters_per_epoch):
    return iters / iters_per_epoch


def split_after_concat(tensor, batch_size):
    """
    In some circumstances we need to concatenate the premises and hypotheses
    tensors to pass them through the same function.  After this has been done
    we may need to split them back up again.  This function does the splitting.
    :param tensor: concatenated tensor of shape [2 * batch_size, a, b], where a
                   is usually num_time_steps and b is usually embed_size
    :param batch_size: the batch_size for the split along the first axis
    :return: two tensors p and h, each of shape [batch_size, a, b]
    """
    p = tf.slice(tensor, [0, 0, 0], [batch_size, -1, -1])
    h = tf.slice(tensor, [batch_size-1, 0, 0], [batch_size, -1, -1])
    return p, h


def unroll_batch(x):
    dims = tf.shape(x)
    unrolled = tf.reshape(x, [dims[0] * dims[1], dims[2]])
    return unrolled
