"""Utilities for TensorFlow."""
import functools
import tensorflow as tf


# UTILITY FUNCTIONS


def clip_gradients(grads_and_vars, norm=3.0, axes=0):
    """Clip gradients.
    Args:
      grads_and_vars: the gradients and variables from the optimizer.
      norm: Float, the value to clip at - default 3.0.
      axes: Integer, the axis along which to clip - default 0.
    Returns:
      List of clipped gradients.
    """
    return [(tf.clip_by_norm(gv[0],
                             clip_norm=norm,
                             axes=axes), gv[1])
            for gv in grads_and_vars
            if gv[0] is not None]


def concat(premises, hypotheses):
    """Concatenate premises and hypotheses along the batch axis.
    Sometimes I need to put both premises and hypotheses tensors
    through the same fully connected layer. We can concatenate
    them along the batch dimension to achieve this.  This also
    applies to other tensors, alphas and betas in the alignment
    model, that are downstream modified tensors relating to
    premises and hypotheses.
    Args:
      premises: tensor in of shape [batch_size, timesteps, embed_size].
      hypotheses: tensor of shape [batch_size, timesteps, embed_size].
    Returns
      Tensor of shape [2 * batch_size, timesteps, embed_size].
    """
    return tf.concat([premises, hypotheses], axis=0)


def fully_connected_with_dropout(inputs,
                                 num_outputs,
                                 activation_fn,
                                 keep_prob):
    fully_connected = tf.contrib.layers.fully_connected(
        inputs=inputs,
        num_outputs=num_outputs,
        activation_fn=activation_fn)
    dropped_out = tf.nn.dropout(
        x=fully_connected,
        keep_prob=keep_prob)
    return dropped_out


def split_after_concat(concatenated, batch_size):
    """Split premises and hypotheses after concatenations.
    Args:
      concatenated: Tensor of shape [2 * batch_size, timesteps, embed_size].
    Returns:
      premises, hypotheses: both tensors of shape [batch_size, timesteps,
        embed_size].
    """
    premises = tf.slice(concatenated, [0, 0, 0], [batch_size, -1, -1])
    hypotheses = tf.slice(
        concatenated, [batch_size-1, 0, 0], [batch_size, -1, -1])
    return premises, hypotheses


# DECORATORS FOR MODELS
# These decorators were designed for TensorFlow by Danijar Hafner:
# https://gist.github.com/danijar/8663d3bbfd586bffecf6a0094cd116f2


def doublewrap(function):
    """Decorator for a decorator.
    Allowing to use the decorator to be used without parentheses if not
    arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """A decorator for functions that define TensorFlow operations.
    The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator
