import tensorflow as tf
from process_data import get_batch_gen


HIDDEN_SIZE = 300
BATCH_SIZE = 3
NUM_ITERS = 50000
WORD_EMBED_DIM = 300
SENT_EMBED_DIM = 100
LONGEST_SENTENCE = 50
ALPHA = 0.01


def data():
    with tf.name_scope('data'):
        # batch size x no. time steps x embedding dimension
        premises = tf.placeholder(tf.float64, [BATCH_SIZE, None, WORD_EMBED_DIM], name='premises')
        hypotheses = tf.placeholder(tf.float64, [BATCH_SIZE, None, WORD_EMBED_DIM], name='hypotheses')
        y = tf.placeholder(tf.float64, [BATCH_SIZE, 1, 3], name='y')
        return premises, hypotheses, y


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
            # output is BATCH_SIZE x LENGTH x WORD_EMBED_DIM


def create_rnns(premises, hypotheses):
    premises_output, _, _ = create_rnn(premises, 'premises')
    hypotheses_output, _, _ = create_rnn(hypotheses, 'hypotheses')
    return premises_output, hypotheses_output


def encode_sentences(sentence_rnn_output, name):
    h1 = tf.contrib.layers.fully_connected(sentence_rnn_output,
                                           SENT_EMBED_DIM,
                                           tf.tanh)
    W_s2 = tf.Variable(tf.random_uniform([1, LONGEST_SENTENCE],
                                         -1.0, 1.0, dtype=tf.float64),
                       name='W_s2')
    encoding = tf.map_fn(lambda h: tf.matmul(W_s2, h),
                         h1,
                         name=name + '_encoding')
    return encoding


def align(premise_output, hypothesis_output):
    similarity = None  # define the similarity function
    alignments = None
    return 4


def classify(encoded_premises, encoded_hypotheses):
    # BATCH_SIZE x 1 x SENT_EMBED_DIM * 2
    concatenated_sents = tf.concat([encoded_premises, encoded_hypotheses],
                                   axis=2,
                                   name='concatenated_sentences')
    # BATCH_SIZE x 1 x SENT_EMBED_DIM * 2
    hidden_layer = tf.contrib.layers.fully_connected(concatenated_sents,
                                                     SENT_EMBED_DIM * 2,
                                                     tf.tanh)
    # BATCH_SIZE x 1 x 3
    logits = tf.contrib.layers.fully_connected(hidden_layer,
                                               3,
                                               None)
    return logits


def loss(logits, y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=y,
                                                                  name='loss'))
    return cost


def optimization(loss):
    optimizer = tf.train.AdamOptimizer(ALPHA).minimize(loss)
    return optimizer


def train(batch_gen):
    for iter in range(NUM_ITERS):  # for batch in read_batch(...) (see CS20SI)
        batch = next(batch_gen)


if __name__ == '__main__':
    premises, hypotheses, y = data()
    premises_output, hypotheses_output, = create_rnns(premises, hypotheses)
    encoded_premises = encode_sentences(premises_output, 'encoded_premises')
    encoded_hypotheses = encode_sentences(hypotheses_output, 'encoded_hypotheses')
    logits = classify(encoded_premises, encoded_hypotheses)
    cost = loss(logits, y)
    optimizer = optimization(cost)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        average_loss = 0.0
        step = 0
        for premises, hypotheses, labels in get_batch_gen(BATCH_SIZE):
            step += 1
            batch_loss, _ = sess.run([cost, optimizer],
                                     {premises: premises,
                                      hypotheses: hypotheses,
                                      y: labels})
            average_loss += batch_loss
            print('Average loss at step %s: %s' % (step, average_loss))
