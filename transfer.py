import tensorflow as tf
import aligned
import model_base
import training
import evaluation
import util
import batching
import mongoi
import numpy as np


def extract_chen_representation():
    print('Extracting Carstens Chen representation...')
    db = mongoi.CarstensDb()
    X = []
    y = []
    print('Working on train...')
    for doc in db.train.all():
        X.append(mongoi.string_to_array(doc['features']))
        y.append(mongoi.string_to_array(doc['sparse_label_encoding']))
    X = np.vstack(X)
    y = np.vstack(y)
    util.save_pickle(X, 'carstens_train_X.pkl')
    util.save_pickle(y, 'carstens_train_y.pkl')
    X = []
    y = []
    print('Working on test...')
    for doc in db.test.all():
        X.append(mongoi.string_to_array(doc['features']))
        y.append(mongoi.string_to_array(doc['sparse_label_encoding']))
    X = np.vstack(X)
    y = np.vstack(y)
    util.save_pickle(X, 'carstens_test_X.pkl')
    util.save_pickle(y, 'carstens_test_y.pkl')
    print('Completed successfully.')


def transfer(db='carstens', train='train', test='test',
             global_step=17168):
    config = model_base.config(learning_rate=5e-4,
                               grad_clip_norm=0.0,
                               p_keep=0.8,
                               p_keep_rnn=1.0,
                               hidden_size=200)
    model = aligned.ChenAlignA(config)

    with tf.Session() as sess:
        print('Random init state:')
        sess.run(tf.global_variables_initializer())
        print('Loss for train: %s' %
              evaluation.loss(model, db, train, sess))
        print('Accuracy for train: %s' %
              evaluation.accuracy(model, db, train, sess))
        print('Loss for test: %s' %
              evaluation.loss(model, db, test, sess))
        print('Accuracy for test: %s' %
              evaluation.accuracy(model, db, test, sess))

        print('Pre-transfer state:')
        util.load_checkpoint_at_step(model_name=model.name,
                                     global_step=global_step,
                                     saver=tf.train.Saver(),
                                     sess=sess)
        model.reset_training_state(sess)
        print('Loss for train: %s' %
              evaluation.loss(model, db, train, sess))
        print('Accuracy for train: %s' %
              evaluation.accuracy(model, db, train, sess))
        print('Loss for test: %s' %
              evaluation.loss(model, db, test, sess))
        print('Accuracy for test: %s' %
              evaluation.accuracy(model, db, test, sess))

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


def get_features():
    config = model_base.config()
    model = aligned.ChenAlignA(config)
    db = mongoi.CarstensDb()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        util.load_checkpoint_at_step(model_name=model.name,
                                     global_step=154512,  # 9th epoch
                                     saver=tf.train.Saver(),
                                     sess=sess)

        # train
        batch_gen = batching.get_batch_gen('carstens', 'train', 1)
        num_iters = batching.num_iters('carstens', 'train', 1)
        for _ in range(num_iters):
            batch = next(batch_gen)
            # [batch_size, 4 * hidden_size]
            features = sess.run(model.aggregate,
                                util.feed_dict(model, batch))
            doc = db.train.get(batch.ids[0])
            doc['features'] = mongoi.array_to_string(features)
            db.train.update(doc)

        # test
        batch_gen = batching.get_batch_gen('carstens', 'test', 1)
        num_iters = batching.num_iters('carstens', 'test', 1)
        for _ in range(num_iters):
            batch = next(batch_gen)
            # [batch_size, 4 * hidden_size]
            features = sess.run(model.aggregate,
                                util.feed_dict(model, batch))
            doc = db.test.get(batch.ids[0])
            doc['features'] = mongoi.array_to_string(features)
            db.test.update(doc)


if __name__ == '__main__':
    config = model_base.config(learning_rate=1e-3,
                               hidden_size=200,
                               p_keep_input=0.8)
    model = aligned.LinearTChen(config)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training.train(model=model,
                       db='carstens',
                       collection='train',
                       tuning_collection='test',
                       num_epochs=200,
                       sess=sess,
                       batch_size=32,
                       load_ckpt=False,
                       save_ckpt=False,
                       transfer=False,
                       batch_gen_fn=batching.get_batch_gen_transfer,
                       feed_dict_fn=util.feed_dict_transfer)
