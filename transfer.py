import tensorflow as tf
import aligned
import model_base
import training
import evaluation
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
                                     global_step=17168,
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
