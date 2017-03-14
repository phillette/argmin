from mongoi import SNLIDb, array_to_string, string_to_array
import spacy
import numpy as np
import itertools


ENCODING_VALUES = {'neutral': 0,
                   'entailment': 1,
                   'contradiction': 2,
                   '-': 0}  # will have to deal with this properly!!!


def sentence_matrix(sentence, nlp):
    doc = nlp(sentence)
    M = np.vstack(list(t.vector.reshape((1, 300)) for t in doc))
    return M


def matrices_into_mongo(nlp, collection):
    db = SNLIDb()
    index = 0
    for doc in db.repository(collection).find_all():
        index += 1
        premise = sentence_matrix(doc['sentence1'], nlp)
        hypothesis = sentence_matrix(doc['sentence2'], nlp)
        db.repository(collection).update_one(doc['_id'], {'premise': array_to_string(premise),
                                                          'hypothesis': array_to_string(hypothesis)})
        print(index)


def encode(label):
    encoding = np.zeros((1, 3), dtype='float64')
    encoding[0, ENCODING_VALUES[label]] = 1
    return encoding


def find_max_length():
    db = SNLIDb()
    max_length = 0
    dev_gen = db.dev.find_all()
    test_gen = db.test.find_all()
    for doc in test_gen:
        if len(doc['sentence1']) > max_length:
            max_length = len(doc['sentence1'])
            print('New max from test = %s' % len(doc['sentence1']))
        if len(doc['sentence2']) > max_length:
            max_length = len(doc['sentence2'])
            print('New max from test = %s' % len(doc['sentence2']))
    for doc in dev_gen:
        if len(doc['sentence1']) > max_length:
            max_length = len(doc['sentence1'])
            print('New max from dev = %s' % len(doc['sentence1']))
        if len(doc['sentence2']) > max_length:
            max_length = len(doc['sentence2'])
            print('New max from dev = %s' % len(doc['sentence2']))
    train_gen = db.train.find_all()
    for doc in train_gen:
        if len(doc['sentence1']) > max_length:
            max_length = len(doc['sentence1'])
            print('New max from train = %s' % len(doc['sentence1']))
        if len(doc['sentence2']) > max_length:
            max_length = len(doc['sentence2'])
            print('New max from train = %s' % len(doc['sentence2']))
    return max_length  # I'm getting 402 - in train; max in dev 300; test 265


class Batch:
    def __init__(self, premises, hypotheses, labels):
        self.premises = premises
        self.hypotheses = hypotheses
        self.labels = labels


def pad_sentence(sentence, desired_length, word_embed_dim):
    original_length = sentence.shape[0]
    if original_length < desired_length:
        sentence = np.concatenate([sentence,
                                   np.zeros((desired_length - original_length,
                                             word_embed_dim))],
                                  axis=0)
    return sentence


def get_batch_gen(batch_size, word_embed_size, collection):
    # each batch is a float64 array of shape: BATCH_SIZE x None x WORD_EMBED_DIM
    db = SNLIDb()
    gen = db.repository(collection).find_all()
    while True:
        premises = []
        hypotheses = []
        labels = []
        max_premise = 0
        max_hypothesis = 0
        # read elements of the batch and determine max size
        for i in range(batch_size):
            doc = next(gen)
            premise = string_to_array(doc['premise'])
            hypothesis = string_to_array(doc['hypothesis'])
            label = encode(doc['gold_label'])
            if premise.shape[0] > max_premise:
                max_premise = premise.shape[0]
            if hypothesis.shape[0] > max_hypothesis:
                max_hypothesis = hypothesis.shape[0]
            premises.append(premise)
            hypotheses.append(hypothesis)
            labels.append(label)
        # pad out matrices
        premises = list(pad_sentence(premise, 402, word_embed_size) for premise in premises)
        hypotheses = list(pad_sentence(hypothesis, 402, word_embed_size) for hypothesis in hypotheses)
        # add third dimension and concatenate
        premises = np.concatenate(list(M[np.newaxis, ...] for M in premises), axis=0)
        hypotheses = np.concatenate(list(M[np.newaxis, ...] for M in hypotheses), axis=0)
        labels = np.concatenate(list(M[np.newaxis, ...] for M in labels), axis=0)
        batch = Batch(premises, hypotheses, labels)
        yield batch


def test_doc():
    with open('data/sco_ind.txt') as f:
        text = f.read()
    paras = text.split('\n')
    sents = itertools.chain(*[para.split('.') for para in text.split('\n') if para != ''])
    sents = [sent.strip() for sent in sents if sent != '']
    nlp = spacy.load('en')
    docs = [nlp(sent) for sent in sents]
    #max_length = 0  # turns out to be 37
    #for doc in docs:
    #    if max_length < len(doc):
    #        max_length = len(doc)
    matrices = [pad_sentence(sentence_matrix(sent, nlp), 402, 300) for sent in sents]
    return matrices


if __name__ == '__main__':
    test_doc()
