import tensorflow as tf


class DropoutConfig:
    """Wrapper for dropout config.

    This class solves the problem of dealing with the
    small details of dropout implementation. We need
    to apply dropout to tensors, for which we need a
    tensor from an op which checks the training state
    and adjusts the dropout rate accordingly. But in
    some places we need the actual value of the keep
    probability. There are also different probabilities
    to apply to different parts of the network and
    these are also wrapped here.

    Attributes:
      raw: dictionary of floats of probabilities indexed
        by type - e.g. raw['ff'] = 0.5.
      ops: dictionary of ops (returning tensors) which wrap
        the keep probabilities in the logic for determining
        whether or not to apply dropout based on whether or
        not the model is in training.
    """

    def __init__(self, config, training_flag):
        """Create a new DropoutConfig object.

        Args:
          config: dictionary of config values - see model_base.config().
        """
        self.raw = {}
        self.ops = {}
        for key in [k for k
                    in config.keys()
                    if k.startswith('p_keep_')]:
            short_key = key.split('p_keep_')[1]
            self.raw[short_key] = config[key]
            self.ops[short_key] = keep_probability(
                p_keep=config[key],
                training_flag=training_flag)


def keep_probability(p_keep, training_flag):
    """Op to get dropout keep probability.

    Args:
      p_keep: float probability of keeping a neuron.
      training_flag: float: if in training 1.0 if not 0.0.

    Returns:
      Tensor (scalar) of keep probability usable for dropout op.
    """
    return tf.subtract(
        tf.cast(1.0, tf.float64),
        tf.multiply(
            training_flag,
            tf.cast(tf.subtract(1.0, p_keep), tf.float64)))
