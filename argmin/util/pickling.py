"""For loading and saving pickles."""
import os
import pickle
from argmin import glovar, errors


PICKLE_DIR = os.path.join(glovar.DATA_DIR, 'pickles')


def load(file_name):
    file_path = os.path.join(PICKLE_DIR, file_name)
    try:
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
            return obj
    except FileNotFoundError:
        raise errors.PickleNotFoundError(file_name)


def save(obj, file_name):
    file_path = os.path.join(PICKLE_DIR, file_name)
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)
