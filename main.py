from models import *
from training import train
from prediction import accuracy


if __name__ == '__main__':
    model = BiRNN(learning_rate=1e-2)
    train(model, 'carstens', 20, load_ckpt=True, save_ckpt=True, transfer=False)
    accuracy(model, 'carstens')  # loading and saving of checkpoints currently not working.

# 54.9306144334 (dev for sure, maybe also test)
