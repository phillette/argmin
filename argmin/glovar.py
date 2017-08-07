"""For global variables."""
import os


WORKING_DIR = os.getcwd()
DATA_DIR = os.path.join(os.getcwd(), 'data')
GLOVE_DIR = os.path.join(DATA_DIR, 'glove', 'glove.840B.300d.txt')
LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2}
MONGO_SERVER = 'localhost'
MONGO_PORT = 27017
REVERSE_LABEL_MAP = {
    0: "entailment",
    1: "neutral",
    2: "contradiction"}
