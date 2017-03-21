from models import *
from training import train
from prediction import accuracy


if __name__ == '__main__':
    model = BiRNN(learning_rate=1e-3)
    train(model, 'train', 2, load_ckpt=False)
    accuracy(model, 'dev', load_ckpt=False)
    accuracy(model, 'test', load_ckpt=False)
