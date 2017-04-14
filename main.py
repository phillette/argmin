from rnn_encoders import *
from training import train
from prediction import accuracy
import tensorflow as tf
from model_base import Config
from aligned import Alignment, AlignmentOld, BiRNNAlignment


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
    # UNREGULARIZED
    # 1e-4 sees successful training for no regularization (5.0 clip) [tanh version]
    # relu version: 1e-3 no good; 1e-4 seems slow; 6e-4 not as good as 1e-4; 3e-4 stuck at 53
    #               training on train @ 1e-4 VERY slow.  After 3 epochs: 71 loss, 44 accuracy.
    # REGULARIZED
    # pki: 0.8; pkf: 0.5 @ 1e-3: Stuck at 55
    # FIXED TO ALL RELU - pumped ff alignment size up to 300
    # pki: 0.8; pkf: 0.5 @ 1e-3: blew the hell up
    #  ""        ""      @ 1e-5: all over the place.  Can't get past 55.
    # back to vanilla unreg but the bias is added (tanh too).
    # 1e-4 will not converge - getting stuck around 52 or so
    # 1e-5 stuck at 56
    # RELU
    #
    config = Config(learning_rate=1e-4,
                    p_keep_input=1.0,
                    p_keep_ff=1.0,
                    grad_clip_norm=5.0,
                    lamda=0.0)
    model = Alignment(config, 300, 100, activation=tf.nn.relu)
    return model


def bi_rnn_aligned():
    """
    Dev results (for quickness)
    ***
    KPRNN  1.0
    KPINP  0.8
    KPFFN  0.5
    GCLIP  5.0
    LAMDA  0.0
    LRATE       1e-2  3e-3  1e-3  9e-4  1e-4  1e-5
    STUCK       *61   *56   *55   *55   *57   *75
    ***
    KPRNN  0.8
    KPINP  0.8
    KPFFN  0.8
    GCLIP  5.0
    LAMDA  0.0
    LRATE       1e-3  1e-4
    STUCK       *55
    ***
    No regularization
    1e-2: stuck @ 59
    1e-3: I seem to remember no love either
    1e-5: ???
    """
    config = Config(learning_rate=1e-4,
                    rnn_size=100,
                    p_keep_rnn=1.0,
                    p_keep_input=1.0,
                    p_keep_ff=1.0,
                    grad_clip_norm=5.0,
                    lamda=0.0)
    model = BiRNNAlignment(config, 2 * config.rnn_size, 100)
    return model


def _train(model, transfer_to_carstens):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train(model, 'snli', 'dev', 20, sess, load_ckpt=False, save_ckpt=True, transfer=False)
        accuracy(model, 'snli', 'train', sess)
        accuracy(model, 'snli', 'dev', sess)
        accuracy(model, 'snli', 'test', sess)
        if transfer_to_carstens:
            model.learning_rate = 1e-6
            #train(model, 'carstens', 'train', 20, sess, load_ckpt=True, save_ckpt=True, transfer=True)
            accuracy(model, 'carstens', 'train', sess, transfer=True)
            accuracy(model, 'carstens', 'test', sess, transfer=True)


if __name__ == '__main__':
    model = aligned()
    transfer_to_carstens = False
    _train(model, transfer_to_carstens)


# 54.9306144334 (dev for sure, maybe also test)
# 73% accuracy on full set of Carstens - looks like it can go higher from the end of training pattern/tendency
# test-train split Carstens: 25 epochs of BiRNN at alpha=1e-5, still dropping at 25. ~68%. Test ~61%.

