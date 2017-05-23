import tensorflow as tf
import aligned
import model_base
import training
import evaluation
import util
import batching
import mongoi
import numpy as np


TRANSFER_MODELS = ['ChenAlignA', 'BiLSTMEnc']
FINAL_PARAM_STEPS = {
    'ChenAlignA': {
        'snli': -1,
        'mnli': -1
    },
    'BiLSTMEnc': {
        'snli': 120176,  # 7 epochs
        'mnli': -1
    }}
COLLECTIONS = {
    'carstens': {
        'train': 'train',
        'tune': 'test'
    }
}


def final_param_path(model_name, transfer_from):
    """Get file path to pre-trained params.

    With the path in hand, call util.load_checkpoint_at_step.

    Args:
      model_name: full model name (defined on the model).
      transfer_from: the db name to transfer from \in {snli, mnli}.

    Returns:
      String of relative file path to the folder with the checkpoint files.

    Raises:
      ValueError if transfer_from or model_name are unexpected.
    """
    if model_name not in TRANSFER_MODELS:
        raise ValueError('Unexpected model_name: %s' % model_name)
    if transfer_from not in ['snli', 'mnli']:
        raise ValueError('Unexpected transfer_from value: %s' % transfer_from)

    path = 'checkpoints/final/%s_%s/%s' % (model_name,
                                           transfer_from,
                                           model_name)
    return path


def transfer_train(model,
                   transfer_from,
                   transfer_to,
                   full_or_linear):
    """Perform transfer learning training.

    Loads the relevant parameters and performs transfer
    training.

    Args:
      model: the model to train.
      transfer_from: the name of the db training was performed on.
      transfer_to: the name of the db to transfer to.
      full_or_linear: whether to use a full classifier or just a
        linear classifier.
    """
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        param_path = final_param_path(model.name, transfer_from)
        step_to_load = FINAL_PARAM_STEPS[model.name][transfer_from]
        util.load_checkpoint_at_step(
            model_name=model.name,
            global_step=step_to_load,
            saver=tf.train.Saver(),
            sess=sess,
            path=param_path)
        # report train and test results with no further training?
        # needs to be a reset of training variables on the model???
        # do the training process...






def extract_chen_representation():
    print('Extracting Carstens Chen representation...')
    db = mongoi.CarstensDb()
    X = []
    y = []
    print('Working on train...')
    for doc in db.train.all():
        X.append(mongoi.string_to_array(doc['features']))
        y.append(doc['sparse_encoding'])
    X = np.vstack(X)
    y = np.vstack(y)
    util.save_pickle(X, 'carstens_train_X.pkl')
    util.save_pickle(y, 'carstens_train_y.pkl')
    X = []
    y = []
    print('Working on test...')
    for doc in db.test.all():
        X.append(mongoi.string_to_array(doc['features']))
        y.append(doc['sparse_encoding'])
    X = np.vstack(X)
    y = np.vstack(y)
    util.save_pickle(X, 'carstens_test_X.pkl')
    util.save_pickle(y, 'carstens_test_y.pkl')
    print('Completed successfully.')


def transfer(db='carstens', train='train', test='test',
             global_step=17168):
    config = model_base.config(learning_rate=5e-4,
                               grad_clip_norm=0.0,
                               p_keep_ff=0.8,
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
