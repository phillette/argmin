import tensorflow as tf
import numpy as np
from util import *
from prediction import accuracy2
from decorators import define_scope


M = 5000              # total number of observations
M_TRAIN = 4500        # number of observations in training set
M_TEST = 500          # number of observations in test set
P_KEEP_INPUT = 0.8         # probability of keeping an input neuron
P_KEEP_HIDDEN = 0.7        # probability of keeping a hidden layer neuron
INPUT_FEATURES = 400  # number of features for input layer (per training observation)
HIDDEN_UNITS = 26     # target number of hidden units (before accounting for dropout)
NUM_LABELS = 10       # number of output labels (the 10 digits)
LEARNING_RATE = 0.01   # optimization global learning rate
BATCH_SIZE = 100
NUM_ITERS = 45
REPORT_EVERY = 5
NUM_ITERS_TEST = 5
NUM_EPOCHS = 40


class Batch2:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y


class BatchGenWrapper:
    def __init__(self, batch_gen_generator, num_iters, report_every):
        self.batch_gen_generator = batch_gen_generator
        self.num_iters = num_iters
        self.report_every = report_every

    def new_batch_generator(self):
        return self.batch_gen_generator()


def accuracy2(model, batch_gen, num_iters, sess):
    average_accuracy = 0
    for iter in range(num_iters):
        batch = next(batch_gen)
        batch_accuracy = sess.run(model.accuracy, feed_dict2(model, batch))
        average_accuracy += batch_accuracy
    print('Accuracy: %s' % (average_accuracy / num_iters))


def train2(model, batch_gen_wrapper, num_epochs, sess,
           load_ckpt=True, save_ckpt=True, write_graph=True, transfer=False):
    # make sure sess.run(tf.global_variables_initializer() has already been run)
    loss_history = []
    accuracy_train_history = []
    if write_graph:
        writer = tf.summary.FileWriter(log_graph_path(model.name), sess.graph)
    saver = tf.train.Saver()
    if load_ckpt:
        load_checkpoint(model, saver, sess, transfer)
    for epoch in range(num_epochs):
        print('Epoch %s' % (epoch + 1))
        batch_gen = batch_gen_wrapper.new_batch_generator()
        cumulative_loss = 0
        cumulative_accuracy = 0.0
        starting_point = model.global_step.eval()
        iteration = model.global_step.eval()
        while iteration < (starting_point + batch_gen_wrapper.num_iters):
            batch = next(batch_gen)
            batch_loss, batch_accuracy, _ = sess.run([model.loss, model.accuracy_train, model.optimize],
                                                     feed_dict2(model, batch))
            cumulative_loss += batch_loss
            cumulative_accuracy += batch_accuracy
            average_loss = cumulative_loss / (iteration + 1)
            average_accuracy = cumulative_accuracy / (iteration + 1)
            loss_history.append(average_loss)
            accuracy_train_history.append(average_accuracy)
            if (iteration + 1) % batch_gen_wrapper.report_every == 0:
                print('Step %s: average loss = %s; average accuracy = %s' % (iteration + 1,
                                                                             average_loss,
                                                                             average_accuracy))
                if save_ckpt:
                    save_checkpoint(model, saver, sess, iteration, transfer)
            iteration += 1
    if write_graph:
        writer.close()
    return loss_history, accuracy_train_history



def label_vector_to_matrix(y):
    """
    Takes a vector of y values with different class labels and creates a matrix of boolean values.
    :param y: vector of y values with different class labels
    :return: a matrix of boolean values
    """
    class_labels = np.unique(y)
    num_classes = len(class_labels)
    Y = np.zeros((y.shape[0], num_classes))
    for k in range(0, num_classes):
        Y[:, k] = (y == class_labels[k])[:, 0]
    return Y


def data():
    X_raw = np.load('data/X.npy')
    y_raw = np.load('data/y.npy')
    Xy = np.concatenate([X_raw, y_raw], axis=1)
    np.random.shuffle(Xy)
    X_train = Xy[:M_TRAIN, :400]
    X_test = Xy[M_TRAIN:, :400]
    y_train = Xy[:M_TRAIN, 400:]
    y_test = Xy[M_TRAIN:, 400:]
    Y_train = label_vector_to_matrix(y_train)
    Y_test = label_vector_to_matrix(y_test)
    return X_train, Y_train, X_test, Y_test


def test_data():
    X = np.load('data/X_test.npy')
    Y = np.load('data/Y_test.npy')
    return X, Y


def batch_generator():
    X = np.load('data/X_train.npy')
    Y = np.load('data/Y_train.npy')
    XY = np.concatenate([X, Y], axis=1)
    indices = list(np.arange(M_TRAIN))
    for _ in range(NUM_ITERS):
        batch_indices = np.random.choice(indices, BATCH_SIZE, replace=False)
        indices = [i for i in indices if i not in batch_indices]
        XY_batch = XY[batch_indices]
        X_batch = XY_batch[:, :400]
        Y_batch = XY_batch[:, 400:]
        yield Batch2(X_batch, Y_batch)


def test_batch_gen():
    X = np.load('data/X_test.npy')
    Y = np.load('data/Y_test.npy')
    for iter in range(NUM_ITERS_TEST):
        start_index = iter * BATCH_SIZE
        end_index = (iter + 1) * BATCH_SIZE
        X_batch = X[start_index:end_index]
        Y_batch = Y[start_index:end_index]
        yield Batch2(X_batch, Y_batch)


class Model:
    def __init__(self):
        self.in_training = True
        self.global_step = tf.Variable(0,
                                       dtype=tf.int32,
                                       trainable=False,
                                       name='global_step')
        self.name = 'MNIST'
        self._data_placeholders
        self._parameters
        self._dropout_vectors
        self._feedforward
        self._feedforward_train
        self._feedforward_test
        self.loss
        self.optimize
        self.accuracy
        self.accuracy_train

    @define_scope('data')
    def _data_placeholders(self):
        self.input_layer_size = INPUT_FEATURES  # 400
        self.hidden_layer_size = 52
        self.X = tf.placeholder(dtype=tf.float32,
                                shape=[None, INPUT_FEATURES],
                                name='X')
        self.X_with_bias = tf.concat([tf.ones(dtype=tf.float32,
                                              shape=[tf.shape(self.X)[0], 1]),
                                      self.X],
                                     axis=1)
        self.Y = tf.placeholder(dtype=tf.float32,
                                shape=[None, NUM_LABELS],
                                name='y')
        return self.X, self.X_with_bias, self.Y

    @define_scope('parameters')
    def _parameters(self):
        self.Theta1 = tf.Variable(tf.random_uniform(shape=[INPUT_FEATURES + 1,
                                                           self.hidden_layer_size],
                                                    minval=-1.0,
                                                    maxval=1.0),
                                  name='Theta1')
        self.Theta2 = tf.Variable(tf.random_uniform(shape=[self.hidden_layer_size + 1,
                                                           NUM_LABELS],
                                                    minval=-1.0,
                                                    maxval=1.0),
                                  name='Theta2')
        return self.Theta1, self.Theta2

    @define_scope('feedforward')
    def _feedforward(self):
        self.X_dropped_out = tf.contrib.layers.dropout(inputs=self.X_with_bias,
                                                       keep_prob=P_KEEP_INPUT,
                                                       is_training=self.in_training)
        self.vanilla_hidden = tf.contrib.layers.fully_connected(inputs=self.X_dropped_out,
                                                                num_outputs=26,
                                                                activation_fn=tf.sigmoid)
        self.hidden_dropped_out = tf.contrib.layers.dropout(inputs=self.vanilla_hidden,
                                                            keep_prob=P_KEEP_HIDDEN,
                                                            is_training=self.in_training)
        self.logits = tf.contrib.layers.fully_connected(inputs=self.hidden_dropped_out,
                                                        num_outputs=10,
                                                        activation_fn=None)
        return self.logits

    @define_scope('dropout_vectors')
    def _dropout_vectors(self):
        self.drop_input = tf.where(condition=tf.random_uniform([1, 401], 0.0, 1.0) > 1 - P_KEEP_INPUT,
                                   x=tf.ones([1, 401]),
                                   y=tf.zeros([1, 401]))
        self.drop_hidden = tf.where(condition=tf.random_uniform([1, 53], 0.0, 1.0) > 1 - P_KEEP_HIDDEN,
                                    x=tf.ones([1, 53]),
                                    y=tf.zeros([1, 53]))
        return self.drop_input, self.drop_hidden

    @define_scope('feedforward_train')
    def _feedforward_train(self):
        self.thinned_input = tf.multiply(self.X_with_bias,  # batch_size x 401
                                         self.drop_input,   # 1 * 401
                                         name='thinned_input')
        self.z1_train = tf.matmul(self.thinned_input,
                                  self.Theta1,
                                  name='z1_train')
        self.a1_train = tf.tanh(self.z1_train, name='a1_train')
        self.a1_train_with_bias = tf.concat([tf.ones(dtype=tf.float32,
                                                     shape=[tf.shape(self.X)[0], 1]),
                                             self.a1_train],
                                            axis=1)
        self.thinned_hidden = tf.multiply(self.drop_hidden,
                                          self.a1_train_with_bias,
                                          name='thinned_hidden')
        self.z2_train = tf.matmul(self.thinned_hidden,
                                  self.Theta2,
                                  name='z2_train')
        return self.z2_train

    @define_scope('feedforward_test')
    def _feedforward_test(self):
        self.z1_test = tf.matmul(self.X_with_bias,
                                 tf.multiply(self.Theta1, P_KEEP_INPUT),
                                 name='z1_test')
        self.a1_test = tf.tanh(self.z1_test,
                               name='a1_test')
        self.a1_test_with_bias = tf.concat([tf.ones(dtype=tf.float32,
                                                    shape=[tf.shape(self.X)[0], 1]),
                                            self.a1_train],
                                           axis=1)
        self.z2_test = tf.matmul(self.a1_test_with_bias,
                                 tf.multiply(self.Theta2, P_KEEP_HIDDEN),
                                 name='z2_test')
        self.a2_test = tf.nn.softmax(self.z2_test, name='a2_test')
        return self.a2_test

    @define_scope
    def loss(self):
        self._loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y,
                                                                           logits=self.z2_train,
                                                                           name='loss'))
        return self._loss

    @define_scope
    def optimize(self):
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        weights = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('weights:0')]
        #grads_and_vars = self.optimizer.compute_gradients(self.loss, weights)
        grads_and_vars = self.optimizer.compute_gradients(self.loss, [self.Theta1, self.Theta2])
        capped_grads_and_vars = [(tf.clip_by_norm(gv[0], clip_norm=5.0, axes=0), gv[1]) for gv in grads_and_vars]
        self._optimize = self.optimizer.apply_gradients(capped_grads_and_vars)
        return self._optimize

    @define_scope
    def accuracy(self):
        self.correct_predictions = tf.equal(tf.argmax(self.a2_test, 1), tf.argmax(self.Y, 1))
        self._accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, tf.float64))
        return self._accuracy

    @define_scope
    def accuracy_train(self):
        self._correct_predictions_train = tf.equal(tf.argmax(tf.nn.softmax(self.z2_train), 1), tf.argmax(self.Y, 1))
        self._accuracy_train = tf.reduce_mean(tf.cast(self._correct_predictions_train, tf.float64))
        return self._accuracy_train


if __name__ == '__main__':
    """
    vanilla test accuracy = 0.834  @ LR = 0.1
    dropout test accuracy = ~0.78  @ LR = 0.1 (higher blows up)
    There is some variability in the droupout performance each time it is run
    probably reflecting the random element.
    It would obviously be more interesting to compare it over time.
    :return:
    """
    model = Model()
    batch_gen_wrapper = BatchGenWrapper(batch_gen_generator=batch_generator,
                                        num_iters=NUM_ITERS,
                                        report_every=REPORT_EVERY)
    X_test, Y_test = test_data()
    with tf.Session() as sess:
        # somehow sort out loss histogram
        sess.run(tf.global_variables_initializer())
        _, _ = train2(model, batch_gen_wrapper, NUM_EPOCHS, sess, load_ckpt=False, save_ckpt=False)
        accuracy2(model, test_batch_gen(), NUM_ITERS_TEST, sess)
