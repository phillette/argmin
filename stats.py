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
LONGEST_SENTENCE_MNLI = 0
LONGEST_SENTENCES = {
    'snli': {
        'train': 402,
        'dev': 300,
        'test': 265
    },
    'mnli': {
        'train': 0,
        'dev_matched': 0,
        'dev_mismatched': 0
    }
}
COLLECTION_SIZE = {
    'snli': {
        'train': 549367,
        'dev': 9842,
        'test': 9824
    },
    'mnli': {
        'train': 392702,
        'dev_matched': 1,
        'dev_mismatched': 1
    },
    'carstens': {
        'all': 4058,
        'train': 3500,
        'test': 558
    }
}
NO_GOLD_LABEL_COUNTS = {
    'snli': {
        'train': 785,
        'dev': 158,
        'test': 176
    },
    'mnli': {
        'train': 0,
        'dev_matched': 185,
        'dev_mismatched': 168
    }
}
SAMPLES_WITH_OOV = {
    'snli': {
        'train': 10297,
        'dev': 161,
        'test': 176
    },
    'mnli': {
        'train': 0,
        'dev_matched': 0,
        'dev_mismatched': 0
    }
}
