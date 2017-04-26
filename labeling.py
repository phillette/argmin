LABEL_TO_ENCODING = {
    'neutral': 0,
    'entailment': 1,
    'contradiction': 2,
    # '-': 0  # used to have this one, but deleted no gold labels so don't want it
}
ENCODING_TO_LABEL = {
    0: 'neutral',
    1: 'entailment',
    2: 'contradiction'
}
