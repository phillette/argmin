import tensorflow as tf
import aligned
import model_base
import prediction
import training
import batching
import stats
import util


db = 'carstens'
train = 'train'
test = 'test'


if __name__ == '__main__':
    config = model_base.config(learning_rate=5e-4,
                               grad_clip_norm=0.0,
                               p_keep=0.8,
                               p_keep_rnn=1.0,
                               hidden_size=200)
    model = aligned.ChenAlignA(config)
    with tf.Session() as sess:
        # first compare a random init state
        print('Random init state:')
        sess.run(tf.global_variables_initializer())
        prediction.accuracy(model=model,
                            db=db,
                            collection=train,
                            sess=sess,
                            batch_size=32)
        prediction.accuracy(model=model,
                            db=db,
                            collection=test,
                            sess=sess,
                            batch_size=32)
        train_batch_gen = batching.get_batch_gen(
            db=db,
            collection=train,
            batch_size=32)
        train_batch = next(train_batch_gen)
        train_loss = sess.run(model.loss,
                              util.feed_dict(model, train_batch))
        print('Loss for training set: %s' % train_loss)
        test_batch_gen = batching.get_batch_gen(
            db=db,
            collection=test,
            batch_size=32)
        test_batch = next(test_batch_gen)
        test_loss = sess.run(model.loss,
                              util.feed_dict(model, test_batch))
        print('Loss for test set: %s' % test_loss)
        prediction.load_ckpt_at_epoch(model=model,
                                      epoch=15,
                                      db_name='snli',
                                      collection_name='train',
                                      batch_size=32,
                                      subset_size=None,
                                      saver=tf.train.Saver(),
                                      sess=sess)
        model.reset_training_state(sess)
        print('Pre-transfer state:')
        prediction.accuracy(model=model,
                            db=db,
                            collection=train,
                            sess=sess,
                            batch_size=32)
        prediction.accuracy(model=model,
                            db=db,
                            collection=test,
                            sess=sess,
                            batch_size=32)
        train_batch_gen = batching.get_batch_gen(
            db=db,
            collection=train,
            batch_size=32)
        train_batch = next(train_batch_gen)
        train_loss = sess.run(model.loss,
                              util.feed_dict(model, train_batch))
        print('Loss for training set: %s' % train_loss)
        test_batch_gen = batching.get_batch_gen(
            db=db,
            collection=test,
            batch_size=32)
        test_batch = next(test_batch_gen)
        test_loss = sess.run(model.loss,
                             util.feed_dict(model, test_batch))
        print('Loss for test set: %s' % test_loss)
        print('Training...')
        training.train(model=model,
                       db=db,
                       collection=train,
                       tuning_collection=test,
                       num_epochs=100,
                       sess=sess,
                       batch_size=4,
                       subset_size=None,
                       load_ckpt=False,
                       save_ckpt=False,
                       transfer=True)
