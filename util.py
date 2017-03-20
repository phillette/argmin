import tensorflow as tf


def add_bias(tensor, dtype=tf.float64, axis=1):
    return tf.concat([tf.ones(dtype=dtype,
                              shape=[tf.shape(tensor)[0], 1]),
                      tensor],
                     axis=axis)


def clip_gradients(grads_and_vars, norm=5.0, axes=0):
    return [(tf.clip_by_norm(gv[0], clip_norm=norm, axes=axes), gv[1]) for gv in grads_and_vars]


def dropout_vector(keep_prob, shape):
    return tf.where(condition=tf.random_uniform(shape, 0.0, 1.0, tf.float64) > 1 - keep_prob,
                    x=tf.ones(shape, tf.float64),
                    y=tf.zeros(shape, tf.float64))
