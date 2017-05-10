LABEL_TO_ENCODING = {
    'neutral': 0,
    'entailment': 1,
    'contradiction': 2,
    'null': 0,  # node
    'n': 0,  # carstens
    's': 1,  # carstens
    'a': 2  # carstens
}
ENCODING_TO_LABEL = {
    0: 'neutral',
    1: 'entailment',
    2: 'contradiction'
}
