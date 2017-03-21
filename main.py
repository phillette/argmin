from models import *
from training import train
from prediction import accuracy


if __name__ == '__main__':
    model = BiRNN(learning_rate=1e-3)
    train(model, 'train', 12, load_ckpt=False)
    accuracy(model, 'train')
    accuracy(model, 'dev')
    accuracy(model, 'test')
