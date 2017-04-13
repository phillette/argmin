from rnn_encoders import *
from training import train
from prediction import accuracy
import tensorflow as tf
from model_base import Config
from aligned import Alignment, BiRNNAlignment


# $ TF_CPP_MIN_LOG_LEVEL=1 python main.py


def bi_rnn():
    model = BiRNN(learning_rate=1e-3)
    return model


def bi_rnn_bowman():
    config = Config(learning_rate=1e-3,
                    p_keep_rnn=1.0,
                    p_keep_input=0.8,
                    p_keep_ff=0.5,
                    grad_clip_norm=5.0,
                    lamda=0.0)
    model = BiRNNBowman(config)
    return model


def aligned():
    config = Config(learning_rate=1e-4,
                    p_keep_rnn=1.0,
                    p_keep_input=1.0,
                    p_keep_ff=1.0,
                    grad_clip_norm=5.0,
                    lamda=0.0)
    model = Alignment(config, config.word_embed_size, 100)
    return model


def bi_rnn_aligned():
    config = Config(learning_rate=1e-4,
                    rnn_size=100,
                    p_keep_rnn=0.5,
                    p_keep_input=0.8,
                    p_keep_ff=0.5,
                    grad_clip_norm=5.0,
                    lamda=0.0)
    model = BiRNNAlignment(config, 2 * config.rnn_size, 100)
    return model


def _train(model, transfer_to_carstens):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train(model, 'snli', 'dev', 30, sess, load_ckpt=False, save_ckpt=True, transfer=False)
        accuracy(model, 'snli', 'train', sess)
        accuracy(model, 'snli', 'dev', sess)
        accuracy(model, 'snli', 'test', sess)
        if transfer_to_carstens:
            model.learning_rate = 1e-6
            #train(model, 'carstens', 'train', 20, sess, load_ckpt=True, save_ckpt=True, transfer=True)
            accuracy(model, 'carstens', 'train', sess, transfer=True)
            accuracy(model, 'carstens', 'test', sess, transfer=True)

if __name__ == '__main__':
    model = bi_rnn_aligned()
    transfer_to_carstens = False
    _train(model, transfer_to_carstens)


# 54.9306144334 (dev for sure, maybe also test)
# 73% accuracy on full set of Carstens - looks like it can go higher from the end of training pattern/tendency
# test-train split Carstens: 25 epochs of BiRNN at alpha=1e-5, still dropping at 25. ~68%. Test ~61%.

