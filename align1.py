import tensorflow as tf


HIDDEN_SIZE = 300
BATCH_SIZE = 3
NUM_ITERS = 50000
EMBEDDING_DIM = 300


def data():
    with tf.name_scope('data'):
        # batch size x no. time steps x embedding dimension
        premises = tf.placeholder(tf.float32, [BATCH_SIZE, None, EMBEDDING_DIM], name='premises')
        hypotheses = tf.placeholder(tf.float32, [BATCH_SIZE, None, EMBEDDING_DIM], name='hypotheses')
        return premises, hypotheses


def create_sentence_rnn(sentences, name):
    with tf.name_scope(name + '_rnn'):
        cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
        initial_state = tf.placeholder_with_default(cell.zero_state(BATCH_SIZE,
                                                                    tf.float32),
                                                    shape=[None, HIDDEN_SIZE],
                                                    name=name + '_initial_state')
        max_sentence_length = tf.reduce_sum(tf.reduce_max(tf.sign(sentences), 2), 1)
        output, output_state = tf.nn.dynamic_rnn(cell,             # cell
                                                 sentences,        # inputs
                                                 max_sentence_length,  # sequence length
                                                 initial_state)    # initial state
        return output, initial_state, output_state


"""
How do I know the shape of the outputs?  In putting the lengths in the tf.nn.dynamic_rnn, am I
telling it how many output units?  I am relying on this being the case in the below.
I could and should find a way to isolate and test this.  Should be doable with a test
function and a feed_dict.  Confirm the behaviour to be expected before building out too far!
"""


def align(premise_output, hypothesis_output):
    similarity = None  # define the similarity function
    alignments = None
    return 4


def train(batch_gen):

    for iter in range(NUM_ITERS):  # for batch in read_batch(...) (see CS20SI)
        batch = next(batch_gen)


if __name__ == '__main__':
    premise, hypothesis = data()
    premise_output, _, _ = create_sentence_rnn(premise, 'premise')
    hypothesis_output, _, _ = create_sentence_rnn(hypothesis, 'hypothesis')
    alignment_output = align(premise_output, hypothesis_output)
