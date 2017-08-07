"""Blocks for TensorFlow models."""
import tensorflow as tf
from coldnet.tensor_flow import util as tf_util


def adam_with_grad_clip(learning_rate, loss, parameters, grad_clip_norm):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # items in grads_and_vars: gv[0] = gradient; gv[1] = variable.
    grads_and_vars = optimizer.compute_gradients(
        loss,
        parameters)
    if grad_clip_norm > 0.0:
        grads_and_vars = tf_util.clip_gradients(
            grads_and_vars=grads_and_vars,
            norm=grad_clip_norm)
    optimizer.apply_gradients(grads_and_vars)


def cross_entropy_with_l2(labels, logits, _lambda, parameters):
    cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=logits,
            name='sparse_softmax_cross_entropy'))
    penalty_term = tf.multiply(
        tf.cast(_lambda, tf.float32),
        sum([tf.nn.l2_loss(p) for p in parameters]),
        name='penalty_term')
    return tf.add(cross_entropy, penalty_term, name='loss')


def embedding_matrix(vocab_length, embed_size, trainable):
    """Get embedding infrastructure.

    Args:
      vocab_length: Integer, how many words in the embedding.
      embed_size: Integer, how long is each embedding.
      trainable: Boolean, whether or not to tune the embeddings.

    Returns:
      embeddings (Variable), embeddings_ph (placeholder), embeddings_init (op
        for assigning the placeholder to the variable).
    """
    embeddings = tf.Variable(
        tf.constant(0.0, shape=[vocab_length, embed_size]),
        trainable=trainable,
        name='embeddings')
    embeddings_ph = tf.placeholder(
        tf.float32, [vocab_length, embed_size])
    # This needs to be called by the trainer
    embeddings_init = embeddings.assign(embeddings_ph)
    return embeddings, embeddings_ph, embeddings_init


def word2vec(embeddings, indices, p_keep):
    vecs = tf.nn.embedding_lookup(embeddings, indices)
    dropped_vecs = tf.nn.dropout(vecs, p_keep)
    return dropped_vecs
