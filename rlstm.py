"""Recursive LSTM classes and functions."""
import tensorflow as tf


class ChildSumTreeLSTMCell(tf.contrib.rnn.BasicLSTMCell):
    """Child-sum tree-LSTM cell.

    This is based on https://arxiv.org/pdf/1503.00075.
    """

    def __init__(self, num_units, keep_prob=1.0):
        """Initialize the cell.

        If we compare to the base class initializer
        (https://github.com/tensorflow/tensorflow/blob/r1.1/
        tensorflow/contrib/rnn/python/ops/core_rnn_cell_impl.py)
        we do not get a number of arguments in the constructor.
        Their defaults on the base class are:
          forget_bias=1.0
          input_size=None (deprecated it seems)
          state_is_tuple=True (sounds like could be deprecated)
          activation=tanh
          reuse=None
        If we want to control any of these in the future we will
        need to add these arguments here.

        Args:
          num_units: int, the number of units in the cell.
          keep_prob: the keep probability for dropout.
        """
        super(ChildSumTreeLSTMCell, self).__init__(num_units)
        self._keep_prob = keep_prob

    def __call__(self, inputs, states, scope=None):
        """Args:
          inputs: tensor, being a vector of size num_units,
            representing the input vector at the current node.
          states: tuple (c, h), being matrices of size
            num_children * num_units, representing the cell
            and hidden states of the child nodes.
          scope: string; the name of the scope.
        """
        with tf.variable_scope(scope or 'child_sum_tree_lstm_cell'):
            c_prev, h_prev = states
            h_bar = tf.reduce_sum(h_prev)
            # combine here for efficiency in the multiplication
            # * we can only combine three here since the forget
            #   gate is a function of h_prev, not h_bar.
            concat = tf.contrib.layers.fully_connected(
                inputs=[inputs, h_bar],
                num_outputs=3 * self._num_units,
                activation_fn=None)
            # i = input_gate, o = output_gate, j = new_input
            i, o, j = tf.split(
                value=concat,
                num_or_size_splits=3,
                axis=1)
            # to deal with the forget gate part I need to declare
            # Variables. How do they do it in the TF code? Is there
            # a cleverer way to do it?
            self.W_xf = tf.Variable(
                initial_value=tf.random_uniform(
                    shape=[self._num_units, self._num_units],
                    minval=-0.1,
                    maxval=0.1,
                    dtype=tf.float32),
                name='W_xf')
            self.W_hf = tf.Variable(
                initial_value=tf.random_uniform(
                    shape=[self._num_units, self._num_units],
                    minval=-0.1,
                    maxval=0.1,
                    dtype=tf.float32),
                name='W_hf')
            # can now calculate the forget gates
            fs = tf.sigmoid(tf.matmul(inputs, self.W_xf)
                            + tf.matmul(h_prev, self.W_hf)
                            + tf.ones(self._num_units))
            c = (tf.sigmoid(i) * tf.tanh(j)
                 + tf.reduce_sum(fs * c_prev))
            h = (tf.sigmoid(o) * tf.tanh(c))

            # still don't get why we return h in both places...
            return h, tf.contrib.rnn.LSTMStateTuple(c, h)


if __name__ == '__main__':
    cell = ChildSumTreeLSTM(300)
