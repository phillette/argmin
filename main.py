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
    config = Config(learning_rate=1e-3,
                    p_keep_input=0.9,
                    p_keep_ff=0.7,
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
    config = Config(learning_rate=1e-3,
                    rnn_size=100,
                    p_keep_rnn=1.0,
                    p_keep_input=1.0,
                    p_keep_ff=1.0,
                    grad_clip_norm=5.0,
                    lamda=0.0)
    model = BiRNNAlignment(config, encoding_size=2 * config.rnn_size, alignment_size=300)
    return model


def _train(model, transfer_to_carstens):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train(model, 'snli', 'dev', 70, sess, load_ckpt=False, save_ckpt=False, transfer=False)
        #accuracy(model, 'snli', 'train', sess)
        accuracy(model, 'snli', 'dev', sess)
        accuracy(model, 'snli', 'test', sess)
        if transfer_to_carstens:
            model.learning_rate = 1e-10
            train(model, 'carstens', 'train', 100, sess, load_ckpt=True, save_ckpt=False, transfer=True)
            accuracy(model, 'carstens', 'train', sess, transfer=True)
            accuracy(model, 'carstens', 'test', sess, transfer=True)


if __name__ == '__main__':
    model = aligned()
    transfer_to_carstens = True
    _train(model, transfer_to_carstens)


"""
p_keep_input; p_keep-ff
NO REG. - train: 89; dev: 67  *20 epochs @ 1e-4
0.95; 0.9 - train: 82; dev: 71 | 78; 46  * but had it really converged???
0.9; 0.8 - 1e-3, 100 epochs, reached about 26 loss, still converging: 88 and 69
0.85; 0.7 - 1e-3, 50 epochs, 35 loss, roughly converged it SEEMED, 80 and 70
0.8; 0.5: train 68
"""


# 54.9306144334 (dev for sure, maybe also test)
# 73% accuracy on full set of Carstens - looks like it can go higher from the end of training pattern/tendency
# test-train split Carstens: 25 epochs of BiRNN at alpha=1e-5, still dropping at 25. ~68%. Test ~61%.

