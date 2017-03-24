from models import *
from training import train
from prediction import accuracy


if __name__ == '__main__':
    model = BiRNN(learning_rate=1e-12)  # originally 1e-3 with good results
    #train(model, 'carstens', 10, load_ckpt=True, transfer=True)
    accuracy(model, 'carstens')
