import tensorflow as tf
import aligned
import model_base
import prediction
import training


if __name__ == '__main__':
    config = model_base.config(learning_rate=1e-7,
                               grad_clip_norm=3.0,
                               p_keep=0.8,
                               p_keep_rnn=1.0,
                               hidden_size=300)
    model = aligned.ChenAlignA(config)
    with tf.Session() as sess:
        prediction.load_ckpt_at_epoch(model=model,
                                      epoch=9,
                                      db_name='snli',
                                      collection_name='train',
                                      batch_size=32,
                                      subset_size=None,
                                      saver=tf.train.Saver(),
                                      sess=sess)
        # reset global_step and other training variables.
        training.train(model=model,
                       db='carstens',
                       collection='train',
                       tuning_collection='test',
                       num_epochs=10,
                       sess=sess,
                       batch_size=32,
                       subset_size=None,
                       load_ckpt=False,
                       save_ckpt=False)
