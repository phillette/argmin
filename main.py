from rnn_encoders import *
from training import train
from prediction import accuracy
import tensorflow as tf
from model_base import Config


# $ TF_CPP_MIN_LOG_LEVEL=1 python main.py


if __name__ == '__main__':
    config = Config(learning_rate=1e-3,
                    p_keep_rnn=1.0,
                    p_keep_input=0.8,
                    p_keep_ff=0.5,
                    grad_clip_norm=5.0,
                    lamda=0.0)
    model = BiRNNBowman(config)
    transfer_to_carstens = False
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train(model, 'snli', 'dev', 20, sess, load_ckpt=True, save_ckpt=True, transfer=False)
        #accuracy(model, 'snli', 'train', sess)
        accuracy(model, 'snli', 'dev', sess)
        accuracy(model, 'snli', 'test', sess)
        if transfer_to_carstens:
            model.learning_rate = 3e-5
            train(model, 'carstens', 'train', 20, sess, load_ckpt=False, save_ckpt=True, transfer=True)
            accuracy(model, 'carstens', 'test', sess)


# 54.9306144334 (dev for sure, maybe also test)
# 73% accuracy on full set of Carstens - looks like it can go higher from the end of training pattern/tendency
# test-train split Carstens: 25 epochs of BiRNN at alpha=1e-5, still dropping at 25. ~68%. Test ~61%.
