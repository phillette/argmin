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


def pad_sentence(sentence, desired_length):
    original_length = sentence.shape[0]
    word_embed_dim = sentence.shape[1]
    if original_length < desired_length:
        sentence = np.concatenate([sentence,
                                   np.zeros((desired_length - original_length,
                                             word_embed_dim))],
                                  axis=0)
    return sentence


def get_batch_gen(batch_size, collection):
    # each batch is a float64 array of shape: BATCH_SIZE x None x WORD_EMBED_DIM
    db = SNLIDb()
    gen = db.repository(collection).find_all()
    while True:
        premises = []
        hypotheses = []
        labels = []
        # read elements of the batch and determine max size
        for i in range(batch_size):
            doc = next(gen)
            premise = string_to_array(doc['premise'])
            hypothesis = string_to_array(doc['hypothesis'])
            label = encode(doc['gold_label'])
            premises.append(premise)
            hypotheses.append(hypothesis)
            labels.append(label)
        batch = Batch(premises, hypotheses, labels)
        pad_out_sentences(batch)
        add_third_dimensions(batch)
        yield batch


def pad_out_sentences(batch, pad_max=True, premise_pad_length=None, hypothesis_pad_length=None):
    # at this stage the premises and hypotheses are still lists of matrices
    if pad_max:
        premise_pad_length = max([premise.shape[0] for premise in batch.premises])
        hypothesis_pad_length = max([hypothesis.shape[0] for hypothesis in batch.hypotheses])
    batch.premises = list(pad_sentence(premise, premise_pad_length) for premise in batch.premises)
    batch.hypotheses = list(pad_sentence(hypothesis, hypothesis_pad_length) for hypothesis in batch.hypotheses)


def add_third_dimensions(batch):
    batch.premises = np.concatenate(list(M[np.newaxis, ...] for M in batch.premises), axis=0)
    batch.hypotheses = np.concatenate(list(M[np.newaxis, ...] for M in batch.hypotheses), axis=0)
    #batch.labels = np.concatenate(list(M[np.newaxis, ...] for M in batch.labels), axis=0)
    batch.labels = np.vstack(batch.labels)


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
    matrices = [pad_sentence(sentence_matrix(sent, nlp), 402) for sent in sents]
    return matrices


if __name__ == '__main__':
    test_doc()
