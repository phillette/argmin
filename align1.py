import tensorflow as tf
from process_data import get_batch_gen
import os


HIDDEN_SIZE = 300
BATCH_SIZE = 100
NUM_ITERS = 5000
WORD_EMBED_DIM = 300
SENT_EMBED_DIM = 100
LONGEST_SENTENCE = 402
ALPHA = 0.0001
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
    predictions = tf.nn.softmax(logits)
    return logits, predictions


def loss(logits, y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=y,
                                                                  name='loss'))
    return cost


def optimization(loss):
    optimizer = tf.train.AdamOptimizer(ALPHA).minimize(loss)
    return optimizer


def train(sess, global_step, cost, optimizer, premises, hypotheses, y):
    with tf.device('/gpu:0'):
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/align1.ckpt'))
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
                saver.save(sess, 'checkpoints/align1.ckpt', iteration)
            iteration += 1


def eval(y_hat, y):
    correct_predictions = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'))
    return correct_predictions, accuracy


def predict(sess, collection, correct_predictions):
    with tf.device('/gpu:0'):
        # init variables
        sess.run(tf.global_variables_initializer())
        # load checkpointed parameter values
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/align1.ckpt'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        # get batches and start predicting
        total_correct = 0
        total_seen = 0
        for batch in get_batch_gen(BATCH_SIZE, WORD_EMBED_DIM, collection):
            number_correct, _ = sess.run(correct_predictions,
                                         {premises: batch.premises,
                                          hypotheses: batch.hypotheses,
                                          y: batch.labels})
            total_correct += number_correct
            total_seen += BATCH_SIZE
            print('Accuracy after %s: %s' % (total_seen, total_correct / total_seen))
        print('Final accuracy: %s' % (total_correct / total_seen))

if __name__ == '__main__':
    premises, hypotheses, y = data()
    premises_output, hypotheses_output, = create_rnns(premises, hypotheses)
    encoded_premises = encode_sentences(premises_output, 'encoded_premises')
    encoded_hypotheses = encode_sentences(hypotheses_output, 'encoded_hypotheses')
    logits, predictions = classify(encoded_premises, encoded_hypotheses)
    cost = loss(logits, y)
    optimizer = optimization(cost)
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    correct_predictions, accuracy = eval(predictions, y)
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        predict(sess, 'test', correct_predictions)
