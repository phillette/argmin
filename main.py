from rnn_encoders import *
from training import train
from prediction import accuracy
import tensorflow as tf
from model_base import Config
import aligned


# $ TF_CPP_MIN_LOG_LEVEL=1 python main.py


def bi_rnn():
    model = BiRNN(learning_rate=1e-3)
    return model


def bi_rnn_bowman():
    config = Config(learning_rate=5e-4,
                    p_keep_rnn=0.8,
                    p_keep_input=1.0,
                    p_keep_ff=0.8,
                    grad_clip_norm=5.0,
                    lamda=0.0)
    model = BiRNNBowman(config)
    return model


def alignment_parikh():
    """ Dev 40ep @ 1e-3: forgot, but roughly 60 at a guess
    70acc 8e-4, ep45(restarted)
    75acc 5e-4, ep45(no restart)
    70ish with extrapolation 1e-4, ep45(no restart)
    """
    config = Config(learning_rate=5e-4,
                    p_keep_ff=0.8,
                    grad_clip_norm=5.0,
                    lamda=0.0,
                    ff_size=200)
    model = aligned.AlignmentParikh(config)
    return model


def alignment_bi_rnn():
    config = Config(learning_rate=5e-4,
                    p_keep_ff=0.8,
                    grad_clip_norm=5.0,
                    lamda=0.0,
                    ff_size=200)
    model = aligned.BiRNNAlignment(config)
    return model


if __name__ == '__main__':
    model = alignment_parikh()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train(model=model,
              db='snli',
              collection='train',
              tuning_collection=None,
              num_epochs=6,
              sess=sess,
              batch_size=20,
              subset_size=4000,
              load_ckpt=True,
              save_ckpt=True,
              transfer=False)
        #accuracy(model, 'snli', 'train', sess, load_ckpt=False)
        #accuracy(model, 'snli', 'dev', sess, load_ckpt=False)
        #accuracy(model, 'snli', 'test', sess, load_ckpt=False)


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

