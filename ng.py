import tensorflow as tf
import numpy as np


M = 5000              # total number of observations
M_TRAIN = 4500        # number of observations in training set
M_TEST = 500          # number of observations in test set
P_INPUT = 0.8         # probability of keeping an input neuron
P_HIDDEN = 0.5        # probability of keeping a hidden layer neuron
INPUT_FEATURES = 400  # number of features for input layer (per training observation)
HIDDEN_UNITS = 26     # target number of hidden units (before accounting for dropout)
NUM_LABELS = 10       # number of output labels (the 10 digits)
LEARNING_RATE = 0.1   # optimization global learning rate


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


class Model:
    def __init__(self):
        self.in_training = True
        self._create_graph()

    def _create_graph(self):
        self._data_placeholders()
        self._parameters()
        self._dropout_vectors()
        self._feedforward()
        self._feedforward_train()
        self._feedforward_test()
        self._loss()
        self._optimize()
        self._accuracy()

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
        self.y = tf.placeholder(dtype=tf.float32,
                                shape=[None, NUM_LABELS],
                                name='y')

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

    def _feedforward(self):
        self.X_dropped_out = tf.contrib.layers.dropout(inputs=self.X_with_bias,
                                                       keep_prob=self.drop_input,
                                                       is_training=self.in_training)
        self.vanilla_hidden = tf.contrib.layers.fully_connected(inputs=self.X_dropped_out,
                                                                num_outputs=26,
                                                                activation_fn=tf.sigmoid)
        self.hidden_dropped_out = tf.contrib.layers.dropout(inputs=self.vanilla_hidden,
                                                            keep_prob=self.drop_hidden,
                                                            is_training=self.in_training)
        self.logits = tf.contrib.layers.fully_connected(inputs=self.hidden_dropped_out,
                                                        num_outputs=10,
                                                        activation_fn=None)

    def _dropout_vectors(self):
        self.drop_input = tf.where(condition=tf.random_uniform([1, 401], 0.0, 1.0) > 0.2,
                                   x=tf.ones([1, 401]),
                                   y=tf.zeros([1, 401]))
        self.drop_hidden = tf.where(condition=tf.random_uniform([1, 53], 0.0, 1.0) > 0.5,
                                    x=tf.ones([1, 53]),
                                    y=tf.zeros([1, 53]))

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

    def _feedforward_test(self):
        self.augmented_input = tf.multiply(P_INPUT,
                                           self.X_with_bias,
                                           name='augmented_input')
        self.z1_test = tf.matmul(self.augmented_input,
                                 self.Theta1,
                                 name='z1_test')
        self.a1_test = tf.tanh(self.z1_test,
                               name='a1_test')
        self.a1_test_with_bias = tf.concat([tf.ones(dtype=tf.float32,
                                                    shape=[tf.shape(self.X)[0], 1]),
                                            self.a1_train],
                                           axis=1)
        self.augmented_hidden = tf.multiply(P_HIDDEN,
                                            self.a1_test_with_bias,
                                            name='augmented_hidden')
        self.z2_test = tf.matmul(self.augmented_hidden,
                                 self.Theta2,
                                 name='z2_test')
        self.a2_test = tf.nn.softmax(self.z2_test, name='a2_test')

    def _loss(self):
        self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.y,
                                                                          logits=self.logits,
                                                                          name='loss'))

    def _optimize(self):
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        weights = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.enswith('weights:0')]
        grads_and_vars = self.optimizer.compute_gradients(self.loss, weights)
        capped_grads_and_vars = [(tf.clip_by_norm(gv[0], clip_norm=5.0, axes=0), gv[1]) for gv in grads_and_vars]
        self.optimize = self.optimizer.apply_gradients(capped_grads_and_vars)

    def _accuracy(self):
        self.correct_predictions = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, tf.float64))


def train():
    """
    vanilla test accuracy = 0.834  @ LR = 0.1
    dropout test accuracy = ~0.78  @ LR = 0.1 (higher blows up)
    There is some variability in the droupout performance each time it is run
    probably reflecting the random element.
    It would obviously be more interesting to compare it over time.
    :return:
    """
    model = Model()
    X_train, Y_train, X_test, Y_test = data()
    batch_size = 100
    num_iters = int(M_TRAIN / batch_size)
    report_every = int(num_iters / 10)
    with tf.Session() as sess:
        # save the graph
        # somehow sort out loss histogram
        sess.run(tf.global_variables_initializer())
        average_loss = 0.0
        average_accuracy = 0.0
        for iter in range(num_iters):
            X_batch = X_train[iter * batch_size:(iter + 1) * batch_size, :]
            y_batch = Y_train[iter * batch_size:(iter + 1) * batch_size, :]
            batch_loss, _, = sess.run([model.loss,
                                       model.optimize],
                                      feed_dict={model.X: X_batch,
                                                 model.y: y_batch})
            average_loss += batch_loss
            if (iter + 1) % report_every == 0:
                print('Iteration %s: loss=%s' % (iter + 1,
                                                 average_loss / (iter + 1)))
        # test set
        test_accuracy = sess.run([model.accuracy],
                                 feed_dict={model.X: X_test,
                                            model.y: Y_test})
        print('Test accuracy: %s' % test_accuracy[0])


if __name__ == '__main__':
    train()
