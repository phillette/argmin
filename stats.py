"""
Notes:
* No gold label observations are definitely removed by default
* At the moment giving the option to remove observations with missing word vectors
"""


"""
Thinking it is best to remove the no gold labels observations:
* train: 550,012 - 785 = 549,227.  This is divisible by 217 2531 times.
* dev:    10,000 - 158 =   9,842.  This is divisible by 259   38 times.
* test:   10,000 - 176 =   9,824.  This is divisible by 307   32 times.
"""


# Basic stats
NUM_LABELS = 3
LONGEST_SENTENCE_SNLI = 403  # 402, but we have prepended NULL to everything now
COLLECTION_SIZE = {
    'snli': {
        'train': 549367,
        'dev': 9842,
        'test': 9824
    },
    'carstens': {
        'all': 4058,
        'train': 3500,
        'test': 558
    }
}

# Stats for batching and training
BATCH_SIZE = {
    'snli': {
        'train': 4,
        'dev': 4,
        'test': 4
    },
    'carstens': {
        'all': 101,
        'train': 100,
        'test': 558
    }
}

NUM_ITERS = {
    'snli': {
        'train': 137342,
        'dev': 1406,
        'test': 2456
    },
    'carstens': {
        'all': 40,
        'train': 35,
        'test': 1
    }
}
REPORT_EVERY = {
    'snli': {
        'train': 5000,
        'dev': 100,
        'test': 100
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

