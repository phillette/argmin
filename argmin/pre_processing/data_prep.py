"""For preparing vocab dicts and embedding matrices."""
import collections
import re
from argmin import glovar
from argmin.util import pickling
import numpy as np


PADDING = "<PAD>"
UNKNOWN = "<UNK>"
LBR = '('
RBR = ')'


def build_embeddings(vocab_dict):
    """Build embeddings with GloVe and random vectors for OOV.

    Make sure GLOVE_PATH is defined in glovar.py.
    Will save the resulting embedding matrix in the pickles directory as
    embeddings.pkl.

    Args:
      vocab_size: Integer, the number of words in the vocab.
    Returns:
      np.ndarray of embeddings.
    """
    vocab_size = max(vocab_dict.values()) + 1
    embed_size = 300
    embeddings = np.random.normal(size=(vocab_size, embed_size))
    # assign zeros to <PAD>
    embeddings[0:2, :] = np.zeros((1, embed_size), dtype='float32')
    with open(glovar.GLOVE_DIR, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            s = line.split()
            if len(s) > 301:  # a hack I needed to use for some reason
                s = [s[0]] + s[-300:]
                assert len(s) == 301
            if s[0] in vocab_dict.keys():
                try:
                    embeddings[vocab_dict[s[0]], :] = np.asarray(s[1:])
                except Exception as e:
                    print(vocab_dict[s[0]])
                    print(len(vocab_dict))
                    print(min(vocab_dict.values()))
                    print(max(vocab_dict.values()))
                    raise Exception('%s, %s:\n%s' % (i, s[0], repr(e)))
    pickling.save(embeddings, 'embeddings.pkl')
    return embeddings


def build_vocab_dict(data, name):
    """Create a vocabulary dictionary.

    The data should be a list of tokens (Strings). It is the responsibility of a
    different function to split the data into individual tokens, and merge them
    all in a single list. The list needn't be a set.

    The vocab dict will be pickled in the pickle directory as:
      vocab_dict_{name}.pkl

    Args:
      data: List of tokens.

    Returns:
      Dictionary: tokens as keys, indices as values.
    """
    print('Building vocab dict...')
    word_counter = collections.Counter()
    for token in data:
        word_counter.update(token.text)
    vocabulary = set([word for word in word_counter])
    vocabulary = list(vocabulary) + [PADDING, UNKNOWN, LBR, RBR]
    vocab_dict = dict(zip(vocabulary, range(len(vocabulary))))
    print('Done. Saving...')
    pickling.save(vocab_dict, 'vocab_dict_%s.pkl' % name)
    return vocab_dict


def tokenize(string, sexpr=False):
    """Get a list of tokens in a string.

    This is designed for NLI data, where we have sentences already parsed. If
    dealing with longer texts, use Spacy.

    Args:
      string: the string to tokenize, usually will be one of the following:
        sentence1
        sentence1_parse
        sentence1_binary_parse
      sexpr: Boolean, if True then the brackets of the sexpr will be treated as
        part of the string and returned as tokens.

    Returns:
      List of tokens (Strings).
    """
    if sexpr:
        rval = string.split()
        return rval
    else:
        string = re.sub(r'\(|\)', '', string)
        rval = string.split()
        return rval
