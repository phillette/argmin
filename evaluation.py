import util
import batching
import tensorflow as tf
import prediction
import stats


def accuracies(model, epoch_to_load, train_db_name, target_db_name):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        iters_per_epoch = batching.num_iters(train_db_name, 'train')
        ckpt_step_to_load = epoch_to_load * iters_per_epoch
        util.load_checkpoint_at_step(
            model_name=model.name,
            global_step=ckpt_step_to_load,
            saver=tf.train.Saver(),
            sess=sess)
        prediction.accuracy(model, target_db_name, 'train', sess)
        for collection in stats.DEV_SETS[target_db_name]:
            prediction.accuracy(model, target_db_name, collection, sess)
        prediction.accuracy(model, target_db_name, 'test', sess)


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
