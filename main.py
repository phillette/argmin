from models import *
from training import train
from prediction import accuracy
import tensorflow as tf


if __name__ == '__main__':
    model = BiRNN(learning_rate=1e-2)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train(model, 'carstens', 20, sess, load_ckpt=True, save_ckpt=True, transfer=False)
        accuracy(model, 'carstens', sess)  # loading and saving of checkpoints currently not working.

# 54.9306144334 (dev for sure, maybe also test)
