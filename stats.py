NUM_LABELS = 3
LONGEST_SENTENCE_SNLI = 403  # 402, but we have prepended NULL to everything
LONGEST_SENTENCE_MNLI = 0
LONGEST_SENTENCES = {
    'snli': {
        'train': 402,
        'dev': 300,
        'test': 265
    },
    'mnli': {
        'train': -1,
        'dev_matched': -1,
        'dev_mismatched': -1
    }
}
COLLECTION_SIZE = {
    'snli': {
        'train': 549367,
        'dev': 9842,
        'test': 9824
    },
    'mnli': {
        'train': 470052,
        'dev_matched': 9815,
        'dev_mismatched': 9832
    },
    'carstens': {
        'all': 4058,
        'train': 3700,
        'test': 358
    },
    'node': {
        'debate_train': 159,
        'debate_test': 161,
        'wiki_train': 228,
        'wiki_test': 224
    },
    'debate': {
        'train': 159,
        'test': 161
    },
    'wiki': {
        'train': 228,
        'test': 224
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
    },
    'carstens': {
        'all': 1353
    }
}
DEV_SETS = {
    'snli': ['dev'],
    'mnli': ['dev_matched', 'dev_mismatched'],
    'carstens': ['all'],
    'debate': [],
    'wiki': []
}
