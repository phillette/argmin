"""Global variables."""
import os


WORKING_DIR = os.getcwd()
DATA_DIR = os.path.join(os.getcwd(), 'data')
GLOVE_DIR = 'd:\\dev\\data\\glove\\'
GLOVE_FILES = {
    300: 'glove.840B.300d.txt'}
FASTTEXT_DIR = 'd:\\dev\\data\\fasttext\\'
FASTTEXT_FILES = {
    'crawl': 'crawl-300d-2M.vec',
    'wiki_news': 'wiki-news-300d-1M.vec',
    'wiki_news_subword': 'wiki-news-300d-1M-subword.vec'}
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
