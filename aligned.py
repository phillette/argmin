import tensorflow as tf
from model_base import Model, fully_connected_with_dropout, Config
from tf_decorators import define_scope
from util import roll_batch, unroll_batch, feed_dict
from batching import get_batch_gen


class Alignment(Model):
    """
    Alignment model without RNN.
    """
    def __init__(self, config, alignment_size):
        Model.__init__(self, config)
        self.name = 'alignment'
        self.alignment_size = alignment_size
        self.align
        self.compare
        #self.aggregate
        #self.logits
        #self.loss
        #self.optimize
        #self.accuracy

    @define_scope
    def align(self):
        W_F = tf.Variable(initial_value=tf.random_uniform(shape=[self.config.word_embed_size,
                                                                 self.alignment_size],
                                                          minval=-1.0,
                                                          maxval=1.0,
                                                          dtype=tf.float64),
                          name='W_F')
        premises_unrolled = unroll_batch(self.X.premises)                     # [batch_size * max_length_p, embed_size]
        hypotheses_unrolled = unroll_batch(self.X.hypotheses)                 # [batch_size * max_length_h, embed_size]
        F_p = tf.tanh(tf.matmul(premises_unrolled, W_F), name='F_p')          # [batch_size * max_length_p, align_size]
        F_h = tf.tanh(tf.matmul(hypotheses_unrolled, W_F), name='F_h')        # [batch_size * max_length_h, align_size]
        F_p_rolled = roll_batch(F_p, tf.shape(self.X.premises))               # [batch_size, max_length_p, align_size]
        F_h_rolled = roll_batch(F_h, tf.shape(self.X.hypotheses))             # [batch_size, max_length_h, align_size]
        eijs = tf.matmul(F_p_rolled,
                         tf.transpose(F_h_rolled,
                                      perm=[0, 2, 1]),
                         name='eijs')                                         # [batch_size, max_length_p, max_length_h]
        eijs_softmaxed = tf.nn.softmax(eijs)                                  # [batch_size, max_length_p, max_length_h]
        betas = tf.matmul(eijs_softmaxed,
                          self.X.hypotheses)                                  # [batch_size, max_length_p, embed_size]
        alphas = tf.matmul(tf.transpose(eijs_softmaxed),
                           self.X.premises)                                   # [batch_size, max_length_h, embed_size]
        return betas, alphas

    @define_scope
    def compare(self):
        betas, alphas = self.align
        W_G = tf.Variable(initial_value=tf.random_uniform(shape=[2 * self.config.word_embed_size,
                                                                 self.config.ff_size],
                                                          minval=-1.0,
                                                          maxval=1.0,
                                                          dtype=tf.float64),
                          name='W_F')
        self.V1_in = tf.concat([self.X.premises, betas], axis=1)
        V2_in = tf.concat([self.X.hypotheses, alphas], axis=1)
        #V1_in_dropped = tf.nn.dropout(self.V1_in, self.config.p_keep_input)
        #V2_in_dropped = tf.nn.dropout(V2_in, self.config.p_keep_input)
        #V1 = tf.tanh(tf.matmul(V1_in_dropped, W_G))
        #V2 = tf.tanh(tf.matmul(V2_in_dropped, W_G))
        #return V1, V2
        return self.V1_in

    @define_scope
    def aggregate(self):
        V1, V2 = self.compare
        v1 = tf.reduce_sum(V1, axis=1)
        v2 = tf.reduce_sum(V2, axis=1)
        concatenated = tf.concat([v1, v2], axis=1)
        return concatenated

    @define_scope
    def logits(self):
        concatenated = self.aggregate
        dropped_input = tf.nn.dropout(concatenated, self.config.p_keep_input)
        a1 = fully_connected_with_dropout(dropped_input,
                                          self.config.ff_size,
                                          tf.tanh,
                                          self.config.p_keep_ff)
        a2 = fully_connected_with_dropout(a1,
                                          self.config.ff_size,
                                          tf.tanh,
                                          self.config.p_keep_ff)
        a3 = tf.contrib.layers.fully_connected(a2, 3, None)
        return a3


if __name__ == '__main__':
    config = Config(learning_rate=1e-3,
                    p_keep_rnn=1.0,
                    p_keep_input=0.8,
                    p_keep_ff=0.5,
                    grad_clip_norm=5.0,
                    lamda=0.0)
    model = Alignment(config, 100)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        batch_gen = get_batch_gen('snli', 'dev')
        batch = next(batch_gen)
        print(sess.run(model.V1_in, feed_dict(model, batch)))
