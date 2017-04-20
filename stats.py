"""
Notes:
* No gold label observations are definitely removed by default
* At the moment giving the option to remove observations with missing word vectors
"""


# Basic stats
NUM_LABELS = 3
LONGEST_SENTENCE_SNLI = 403  # 402, but we have prepended NULL to everything now


# Stats for batching and training
BATCH_SIZE = {
    'snli': {
        'train': 217,
        'dev': 259,
        'test': 307
    },
    'carstens': {
        'all': 101,
        'train': 100,
        'test': 558
    }
}
COLLECTION_SIZE = {
    'snli': {
        'train': 550012,
        'dev': 9842,
        'test': 9824
    },
    'carstens': {
        'all': 4058,
        'train': 3500,
        'test': 558
    }
}
NUM_ITERS = {
    'snli': {
        'train': 2531,
        'dev': 38,
        'test': 32
    },
    'carstens': {
        'all': 40,
        'train': 35,
        'test': 1
    }
}
REPORT_EVERY = {
    'snli': {
        'train': 250,
        'dev': 5,
        'test': 6
    },
    'carstens': {
        'all': 4,
        'train': 5,
        'test': 1
    }
}


# For the old buffer-style batch generator - probably move it to there?
BUFFER_FACTORS = {
    'snli': {'train': 4,
             'dev': 4,
             'test': 4},
    'carstens': {'all': 4,
                 'train': 35,
                 'test': 1}
}

# Encoding mappings for labels
ENCODING_TO_LABEL = {0: 'neutral',
                     1: 'entailment',
                     2: 'contradiction'}
LABEL_TO_ENCODING = {'neutral': 0,
                     'entailment': 1,
                     'contradiction': 2,
                     '-': 0}  # this is an issue
