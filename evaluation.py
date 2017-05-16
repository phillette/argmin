import util
import batching


def accuracy(model, db, collection, sess):
    accumulated_accuracy = 0.0
    batch_size = batching.get_batch_size(db, collection)
    batch_gen = batching.get_batch_gen(db, collection, batch_size)
    num_iters = batching.num_iters(db, collection, batch_size)
    for _ in range(num_iters):
        batch = next(batch_gen)
        accuracy = sess.run(model.accuracy,
                            util.feed_dict(model, batch))
        accumulated_accuracy += accuracy
    return accumulated_accuracy / num_iters


def loss(model, db, collection, sess):
    accumulated_loss = 0.0
    batch_size = batching.get_batch_size(db, collection)
    batch_gen = batching.get_batch_gen(db, collection, batch_size)
    num_iters = batching.num_iters(db, collection, batch_size)
    for _ in range(num_iters):
        batch = next(batch_gen)
        loss = sess.run(model.loss,
                        util.feed_dict(model, batch))
        accumulated_loss += loss
    return accumulated_loss / batch_size / num_iters
