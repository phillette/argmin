import tensorflow as tf
import stats
import decorators
import util


class X:
    def __init__(self, premises, hypotheses):
        self.premises = premises
        self.hypotheses = hypotheses


def data_placeholders(word_embed_size):
    premises = tf.placeholder(tf.float64,
                              [None,
                               None,
                               word_embed_size],
                              name='premises')
    hypotheses = tf.placeholder(tf.float64,
                                [None,
                                 None,
                                 word_embed_size],
                                name='hypotheses')
    Y = tf.placeholder(tf.float64,
                       [None, 3],
                       name='y')
    return X(premises, hypotheses), Y


def fully_connected_with_dropout(inputs, num_outputs, activation_fn, p_keep):
    fully_connected = tf.contrib.layers.fully_connected(inputs,
                                                        num_outputs,
                                                        activation_fn)
    dropped_out = tf.nn.dropout(fully_connected, p_keep)
    return dropped_out


class Config:
    def __init__(self,
                 word_embed_size=300,
                 learning_rate=5e-4,
                 time_steps=stats.LONGEST_SENTENCE_SNLI,
                 grad_clip_norm=5.0,
                 hidden_size=100,
                 rnn_size=300,
                 ff_size=200,
                 lamda=0.0,
                 p_keep_input=0.8,
                 p_keep_rnn=0.5,
                 p_keep_ff=0.8):
        self.word_embed_size = word_embed_size
        self.learning_rate = learning_rate
        self.time_steps = time_steps
        self.grad_clip_norm = grad_clip_norm
        self.hidden_size = hidden_size
        self.rnn_size = rnn_size
        self.ff_size = ff_size
        self.lamda = lamda
        self.p_keep_input = p_keep_input
        self.p_keep_rnn = p_keep_rnn
        self.p_keep_ff = p_keep_ff


def base_config(embed_size=300,
                learning_rate=1e-3,
                grad_clip_norm=0.0,
                hidden_size=200,
                lamda=0.0,
                p_keep=0.8):
    return {
        'embed_size': embed_size,
        'learning_rate': learning_rate,
        'grad_clip_norm': grad_clip_norm,
        'hidden_size': hidden_size,
        'lambda': lamda,
        'p_keep': p_keep
    }


class Model:
    def __init__(self, config):
        self.config = config
        self.embed_size = config['embed_size']
        self.learning_rate = config['learning_rate']
        self.grad_clip_norm = config['grad_clip_norm']
        self.hidden_size = config['hidden_size']
        self.lamda = config['lambda']
        self.p_keep = config['p_keep']
        self.global_step = tf.Variable(initial_value=0,
                                       dtype=tf.int32,
                                       trainable=False,
                                       name='global_step')
        self.global_epoch = tf.Variable(initial_value=0,
                                        dtype=tf.int32,
                                        trainable=False,
                                        name='global_epoch')
        self.accumulated_loss = tf.Variable(initial_value=0,
                                            dtype=tf.float32,
                                            trainable=False,
                                            name='average_loss')
        self.accumulated_accuracy = tf.Variable(initial_value=0,
                                                dtype=tf.float32,
                                                trainable=False,
                                                name='average_accuracy')
        self.training_history_id = tf.Variable(initial_value=-1,
                                               dtype=tf.int32,
                                               trainable=False,
                                               name='training_history_id')
        self.data
        self.batch_size
        self.batch_timesteps

    @decorators.define_scope
    def accuracy(self):
        return tf.reduce_mean(tf.cast(self.correct_predictions, tf.float64))

    @decorators.define_scope
    def batch_size(self):
        return tf.shape(self.X.premises)[0]

    @decorators.define_scope
    def batch_timesteps(self):
        return tf.shape(self.X.premises)[1]

    @decorators.define_scope
    def confidences(self):
        return tf.reduce_max(self.logits, axis=1)

    @decorators.define_scope
    def correct_predictions(self):
        return tf.equal(self.predicted_labels, tf.argmax(self.Y, axis=1))

    @decorators.define_scope
    def data(self):
        self.X, self.Y = data_placeholders(self.embed_size)
        return self.X, self.Y

    @decorators.define_scope
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.loss,
                                                     self._all_weights())
        if self.grad_clip_norm > 0.0:
            grads_and_vars = util.clip_gradients(grads_and_vars,
                                                 norm=self.grad_clip_norm)
        return optimizer.apply_gradients(grads_and_vars)

    @decorators.define_scope
    def loss(self):
        cross_entropy = tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=self.Y,
                logits=self.logits,
                name='softmax_cross_entropy'))
        penalty_term = tf.multiply(
            tf.cast(self.lamda, tf.float64),
            sum([tf.nn.l2_loss(w) for w in self._all_weights()]),
            name='penalty_term')
        return tf.add(cross_entropy, penalty_term, name='loss')

    @decorators.define_scope
    def predicted_labels(self):
        return tf.argmax(self.logits, axis=1)

    @decorators.define_scope
    def summary(self):
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.histogram('histogram_loss', self.loss)
        return tf.summary.merge_all()

    def _weights(self, scope):
        vars = tf.global_variables()
        weights_name = '%s/weights:0' % scope
        if weights_name not in [v.name for v in vars]:
            raise Exception('Could not find weights with name %s'
                            % weights_name)
        return next(v for v in vars if v.name == weights_name)

    def _all_weights(self):
        return [v for
                v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                if v.name.endswith('weights:0')]


if __name__ == '__main__':
    a = tf.placeholder(tf.float32, [None, 3, 3])
    import numpy as np
    _a = np.random.rand(2, 3, 3)
    fd = {a: _a}
    size = tf.shape(a)[0]
    op = tf.square(size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(op, fd))
