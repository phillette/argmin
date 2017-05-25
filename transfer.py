"""Code for transfer learning."""
import tensorflow as tf
import aligned
import model_base
import training
import evaluation
import util
import batching
import mongoi
import numpy as np
import prediction
import stats
import rnn_encoders


"""The transfer process.

* First pickle the params...

* FOR EACH TARGET DB
1) initial_accuracies(transfer_from='snli')
2) initial_accuracies(transfer_from='mnli')
3) transfer_train(transfer_from='snli', linear=False)
4) transfer_train(transfer_from='snli', linear=True)
5) transfer_train(transfer_from='mnli', linear=False)
6) transfer_train(transfer_from='mnli', linear=True)
7) final_accuracies()
"""


TRANSFER_MODELS = ['ChenAlignA', 'BiLSTMEnc']
FINAL_PARAM_STEPS = {
    'ChenAlignA': {
        'snli': -1,
        'mnli': -1
    },
    'BiLSTMEnc': {
        'snli': 257520,  # 15 epochs
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


def initial_accuracies(model,
                       transfer_from,
                       transfer_to):
    """Get initial accuracies from learned params.

    Args:
      model: the model to evaluate.
      transfer_from: the name of the db training was performed on.
      transfer_to: the name of the db to transfer to.
    """
    print('Determining initial transfer accuracies '
          'for model %s train on %s with target %s...'
          % (model.name, transfer_from, transfer_to))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        load_model_from_pickles(
            model,
            transfer_from,
            sess,
            excluded_scopes=['linear_logits'])
        prediction.accuracy(model, transfer_to, 'train', sess)
        for collection in stats.DEV_SETS[transfer_to]:
            prediction.accuracy(model, transfer_to, collection, sess)
        prediction.accuracy(model, transfer_to, 'test', sess)


def load_model(model, transfer_from, sess):
    """Initialize the model parameters.

    Args:
      model: the model to load.
      transfer_from: the name of the db training was performed on.
    """
    param_path = final_param_path(model.name, transfer_from)
    step_to_load = FINAL_PARAM_STEPS[model.name][transfer_from]
    util.load_checkpoint_at_step(
        model_name=model.name,
        global_step=step_to_load,
        saver=tf.train.Saver(),
        sess=sess,
        path=param_path)


def load_model_from_pickles(model, transfer_from, sess, excluded_scopes=[]):
    for weight in model._all_weights():
        if not param_excluded(weight.name, excluded_scopes):
            params = util.load_pickle(pickle_name(model.name,
                                                  transfer_from,
                                                  weight.name))
            sess.run(tf.assign(weight, params))
    for bias in model._all_biases():
        if not param_excluded(bias.name, excluded_scopes):
            params = util.load_pickle(pickle_name(model.name,
                                                  transfer_from,
                                                  bias.name))
            sess.run(tf.assign(bias, params))


def param_excluded(name, excluded_scopes):
    scope = name.split('/')[0]
    return scope in excluded_scopes


def pickle_name(model_name, transfer_from, weight_name):
    dir = '%s-%s/' % (model_name, transfer_from)
    name = weight_name.replace('/', '-')
    name = name.replace(':', '--')
    return dir + name + '.pkl'


def pickle_params(model, transfer_from):
    with tf.Session() as sess:
        load_model(model, transfer_from, sess)
        for weight in model._all_weights():
            w = sess.run(weight)
            util.save_pickle(w, pickle_name(model.name,
                                            transfer_from,
                                            weight.name))
        for bias in model._all_biases():
            b = sess.run(bias)
            util.save_pickle(b, pickle_name(model.name,
                                            transfer_from,
                                            bias.name))


def transfer_train(model,
                   transfer_from,
                   transfer_to,
                   load_ckpt=False):
    """Perform transfer learning training.

    Loads the relevant parameters and performs transfer
    training.

    Learning rates should already be set on the model passed.

    Args:
      model: the model to train.
      transfer_from: the name of the db training was performed on.
      transfer_to: the name of the db to transfer to.
      linear: boolean indicatig whether to use a full classifier (False)
        or just a linear classifier (True).
    """
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        load_model_from_pickles(
            model,
            transfer_from,
            sess,
            excluded_scopes=['linear_logits'])
        model.reset_training_state(sess)
        training.train(
            model=model,
            db=transfer_to,
            collection='train',
            num_epochs=100,
            sess=sess,
            batch_size=4,
            tuning_collection='test',
            load_ckpt=load_ckpt,
            save_ckpt=True,
            transfer=True)


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
    config = model_base.config(  # the config should be loaded with ckpt...
        p_keep_ff=0.5,
        p_keep_rnn=0.8,
        p_keep_input=0.8,
        hidden_size=200,
        representation_learning_rate=5e-6,
        classifier_learning_rate=5e-5,
        linear_classifier_learning_rate=1e-3,
        linear_logits_output=3)
    model = rnn_encoders.BiLSTMEncoder(config)
    target = 'carstens'
    #transfer.initial_accuracies(
    #    model=model,
    #    transfer_from='snli',
    #    transfer_to=target)
    #transfer.initial_accuracies(
    #    model=model,
    #    transfer_from='mnli',
    #    transfer_to=target)
    transfer_train(
        model=model,
        transfer_from='snli',
        transfer_to=target,
        load_ckpt=True)
