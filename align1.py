import tensorflow as tf


HIDDEN_SIZE = 300
BATCH_SIZE = 3
NUM_ITERS = 50000
WORD_EMBED_DIM = 300
SENT_EMBED_DIM = 100
LONGEST_SENTENCE = 50


def data():
    with tf.name_scope('data'):
        # batch size x no. time steps x embedding dimension
        premises = tf.placeholder(tf.float64, [BATCH_SIZE, None, WORD_EMBED_DIM], name='premises')
        hypotheses = tf.placeholder(tf.float64, [BATCH_SIZE, None, WORD_EMBED_DIM], name='hypotheses')
        return premises, hypotheses


def create_rnn(sentences, name):
    with tf.variable_scope(name + '_rnn_scope'):
        with tf.name_scope(name + '_rnn'):
            cell = tf.contrib.rnn.GRUCell(HIDDEN_SIZE)
            initial_state = tf.placeholder_with_default(cell.zero_state(BATCH_SIZE,
                                                                        tf.float64),
                                                        shape=[None, HIDDEN_SIZE],
                                                        name=name + '_initial_state')
            length = tf.reduce_sum(tf.reduce_max(tf.sign(sentences), 2), 1)
            #length = tf.multiply(tf.ones([BATCH_SIZE]), 50)
            output, output_state = tf.nn.dynamic_rnn(cell,              # cell
                                                     sentences,         # inputs
                                                     length,            # sequence length
                                                     initial_state)     # initial state
            return output, initial_state, output_state


def create_rnns(premises, hypotheses):
    premises_output, _, _ = create_rnn(premises, 'premises')
    hypotheses_output, _, _ = create_rnn(hypotheses, 'hypotheses')
    return premises_output, hypotheses_output


def encode_sentences(sentence_rnn_output, name):
    with tf.variable_scope(name + '_encode_scope'):
        W_s1 = tf.Variable(tf.random_uniform([WORD_EMBED_DIM, SENT_EMBED_DIM],
                                             -1.0, 1.0, dtype=tf.float64),
                           name='W_s1')
        W_s2 = tf.Variable(tf.random_uniform([1, LONGEST_SENTENCE],
                                             -1.0, 1.0, dtype=tf.float64),
                           name='W_s2')
        encode1 = tf.map_fn(lambda h: tf.tanh(tf.matmul(h, W_s1)),
                            sentence_rnn_output,
                            name='encode1')
        encoding = tf.map_fn(lambda h: tf.tanh(tf.matmul(W_s2, h)),
                             encode1,
                             name=name + '_encoding')
        return encoding


def align(premise_output, hypothesis_output):
    similarity = None  # define the similarity function
    alignments = None
    return 4


def classify(encoded_premises, encoded_hypotheses):
    # BATCH_SIZE x 1 x 200
    concatenated_sentences = tf.concat([encoded_premises, encoded_hypotheses],
                                       axis=2,
                                       name='concatenated_sentences')
    concatenated_sentences_with_bias = tf.concat([tf.ones([BATCH_SIZE, 1, 1],
                                                          dtype=tf.float64),
                                                  concatenated_sentences],
                                                 axis=2)
    W_h1 = tf.Variable(tf.random_uniform([BATCH_SIZE, 2 * SENT_EMBED_DIM + 1]))
    hidden_layer_z = tf.placeholder(tf.float64,
                                    [BATCH_SIZE, 1, 200],
                                    name='hidden_layer_z')



def train(batch_gen):

    for iter in range(NUM_ITERS):  # for batch in read_batch(...) (see CS20SI)
        batch = next(batch_gen)


if __name__ == '__main__':
    premise, hypothesis = data()
    premise_output, _, _ = create_sentence_rnn(premise, 'premise')
    hypothesis_output, _, _ = create_sentence_rnn(hypothesis, 'hypothesis')
    alignment_output = align(premise_output, hypothesis_output)
