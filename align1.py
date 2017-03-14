import tensorflow as tf
from process_data import get_batch_gen
import os


HIDDEN_SIZE = 300
BATCH_SIZE = 100
NUM_ITERS = 50000
WORD_EMBED_DIM = 300
SENT_EMBED_DIM = 100
LONGEST_SENTENCE = 402
ALPHA = 1e-2
REPORT_EVERY = 100


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
            length = tf.reduce_sum(tf.reduce_max(tf.sign(sentences), 2), 1)  # need to confirm this works
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
    with tf.device('/gpu:0'):
        premises, hypotheses, y = data()
        premises_output, hypotheses_output, = create_rnns(premises, hypotheses)
        encoded_premises = encode_sentences(premises_output, 'encoded_premises')
        encoded_hypotheses = encode_sentences(hypotheses_output, 'encoded_hypotheses')
        logits = classify(encoded_premises, encoded_hypotheses)
        cost = loss(logits, y)
        optimizer = optimization(cost)
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        saver = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            writer = tf.summary.FileWriter('graphs', sess.graph)
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/align1/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            average_loss = 0.0
            iteration = global_step.eval()
            for batch in get_batch_gen(BATCH_SIZE, WORD_EMBED_DIM, 'train'):
                batch_loss, _ = sess.run([cost, optimizer],
                                         {premises: batch.premises,
                                          hypotheses: batch.hypotheses,
                                          y: batch.labels})
                average_loss += batch_loss
                if (iteration + 1) % REPORT_EVERY == 0:
                    print('Average loss at step %s: %s' % (iteration + 1,
                                                           average_loss / (iteration + 1)))
                    saver.save(sess, 'checkpoints/align1', iteration)
                iteration += 1
        # need a test of accuracy here...
