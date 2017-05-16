import tensorflow as tf
import decorators
import util


def fully_connected_with_dropout(inputs,
                                 num_outputs,
                                 activation_fn,
                                 p_keep):
    fully_connected = tf.contrib.layers.fully_connected(
        inputs,
        num_outputs,
        activation_fn)
    dropped_out = tf.nn.dropout(fully_connected, p_keep)
    return dropped_out


def config(embed_size=300,
           learning_rate=1e-3,
           grad_clip_norm=0.0,
           hidden_size=200,
           lamda=0.0,
           p_keep=0.8,
           p_keep_rnn=1.0):
    return {
        'embed_size': embed_size,
        'learning_rate': learning_rate,
        'grad_clip_norm': grad_clip_norm,
        'hidden_size': hidden_size,
        'lambda': lamda,
        'p_keep': p_keep,
        'p_keep_rnn': p_keep_rnn
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
        self.weights_to_scale_factor = {}
        self._training_variables()
        self._training_placeholders()
        self._training_ops()
        self.premises
        self.hypotheses
        self.Y
        self.batch_size
        self.batch_timesteps

    @decorators.define_scope
    def accuracy(self):
        return tf.reduce_mean(tf.cast(self.correct_predictions, tf.float64))

    @decorators.define_scope
    def batch_size(self):
        return tf.shape(self.premises)[0]

    @decorators.define_scope
    def batch_timesteps(self):
        return tf.shape(self.premises)[1]

    @decorators.define_scope
    def confidences(self):
        return tf.reduce_max(self.logits, axis=1)

    @decorators.define_scope
    def correct_predictions(self):
        return tf.equal(self.predicted_labels, tf.argmax(self.Y, axis=1))

    @decorators.define_scope
    def hypotheses(self):
        return tf.placeholder(
            tf.float64,
            [None, None, self.embed_size],
            name='hypotheses')

    @decorators.define_scope
    def loss(self):
        cross_entropy = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.argmax(self.Y, axis=1),
                logits=self.logits,
                name='softmax_cross_entropy'))
        penalty_term = tf.multiply(
            tf.cast(self.lamda, tf.float64),
            sum([tf.nn.l2_loss(w) for w in self._all_weights()]),
            name='penalty_term')
        return tf.add(cross_entropy, penalty_term, name='loss')

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
    def optimize_transfer(self):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        weights_to_optimize = self._transfer_training_weights()
        grads_and_vars = optimizer.compute_gradients(
            self.loss,
            weights_to_optimize)
        if self.grad_clip_norm > 0.0:
            grads_and_vars = util.clip_gradients(grads_and_vars,
                                                 norm=self.grad_clip_norm)
        return optimizer.apply_gradients(grads_and_vars)

    @decorators.define_scope
    def predicted_labels(self):
        return tf.argmax(self.logits, axis=1)

    @decorators.define_scope
    def premises(self):
        return tf.placeholder(
            tf.float64,
            [None, None, self.embed_size],
            name='premises')

    def scale_weights(self, sess):
        """Scale dropped out weights for prediction."""
        for weight in self.weights_to_scale_factor.keys():
            sess.run(tf.multiply(
                weight,
                self.weights_to_scale_factor[weight]))

    @decorators.define_scope
    def summary(self):
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.histogram('histogram_loss', self.loss)
        return tf.summary.merge_all()

    @decorators.define_scope
    def Y(self):
        return tf.placeholder(
            tf.float64,
            [None, 3],
            name='y')

    def _all_weights(self):
        return [v for
                v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                if v.name.endswith('weights:0')]

    def _init_backend(self):
        self.loss
        self.optimize
        self.optimize_transfer
        self.predicted_labels
        self.correct_predictions
        self.accuracy
        self.confidences
        self.summary

    def _training_ops(self):
        self.update_epoch = tf.assign(
            self.global_epoch,
            self.new_global_epoch)
        self.update_iter = tf.assign(
            self.global_step,
            self.new_global_step)
        self.update_loss = tf.assign(
            self.accumulated_loss,
            self.new_accumulated_loss)
        self.update_accuracy = tf.assign(
            self.accumulated_accuracy,
            self.new_accumulated_accuracy)
        self.update_tuning_iter = tf.assign(
            self.tuning_iter,
            self.new_tuning_iter)
        self.update_tuning_accuracy = tf.assign(
            self.accumulated_tuning_accuracy,
            self.new_accumulated_tuning_accuracy)
        self.set_training_history_id = tf.assign(
            self.training_history_id,
            self.new_training_history_id)

    def _training_placeholders(self):
        self.new_global_epoch = tf.placeholder(tf.int32)
        self.new_global_step = tf.placeholder(tf.int32)
        self.new_accumulated_loss = tf.placeholder(tf.float32)
        self.new_accumulated_accuracy = tf.placeholder(tf.float32)
        self.new_accumulated_tuning_accuracy = tf.placeholder(tf.float32)
        self.new_tuning_iter = tf.placeholder(tf.int32)
        self.new_training_history_id = tf.placeholder(tf.int32)

    def _training_variables(self):
        self.global_step = tf.Variable(
            initial_value=0,
            dtype=tf.int32,
            trainable=False,
            name='global_step')
        self.global_epoch = tf.Variable(
            initial_value=0,
            dtype=tf.int32,
            trainable=False,
            name='global_epoch')
        self.accumulated_loss = tf.Variable(
            initial_value=0,
            dtype=tf.float32,
            trainable=False,
            name='accumulated_loss')
        self.accumulated_accuracy = tf.Variable(
            initial_value=0,
            dtype=tf.float32,
            trainable=False,
            name='accumulated_accuracy')
        self.tuning_iter = tf.Variable(
            initial_value=0,
            dtype=tf.int32,
            trainable=False,
            name='tuning_iter'
        )
        self.accumulated_tuning_accuracy = tf.Variable(
            initial_value=0.0,
            dtype=tf.float32,
            trainable=False,
            name='accumulated_tuning_accuracy'
        )
        self.training_history_id = tf.Variable(
            initial_value=-1,
            dtype=tf.int32,
            trainable=False,
            name='training_history_id')

    def _transfer_training_weights(self):
        return self._all_weights()

    def _weights(self, scope):
        vars = tf.global_variables()
        weights_name = '%s/weights:0' % scope
        if weights_name not in [v.name for v in vars]:
            raise Exception('Could not find weights with name %s'
                            % weights_name)
        return next(v for v in vars if v.name == weights_name)

    def reset_training_state(self, sess):
        sess.run([self.update_epoch,
                  self.update_iter,
                  self.update_loss,
                  self.update_accuracy,
                  self.update_tuning_iter,
                  self.update_tuning_accuracy,
                  self.set_training_history_id],
                 {self.new_global_epoch: 0,
                  self.new_global_step: 0,
                  self.new_accumulated_loss: 0.0,
                  self.new_accumulated_accuracy: 0.0,
                  self.new_tuning_iter: 0,
                  self.new_accumulated_tuning_accuracy: 0.0,
                  self.new_training_history_id: -1})
