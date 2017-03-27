from models import *
from training import train
from prediction import accuracy


if __name__ == '__main__':
    model = LSTMEncoder(learning_rate=1e-1,
                        p_keep_input=0.8,
                        p_keep_hidden=0.5,
                        grad_norm=5.0)
    train(model, 'dev', 10, load_ckpt=True, transfer=False)
    accuracy(model, 'train')
    accuracy(model, 'dev')
    accuracy(model, 'test')

# 54.9306144334 (dev for sure, maybe also test)
