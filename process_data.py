from mongoi import SNLIDb, array_to_string, string_to_array, Carstens
import spacy
import numpy as np
import itertools
import pandas as pd


"""
Note that to split the Carstens data into train and test, I used a variant on this code:
http://stackoverflow.com/questions/27039083/mongodb-move-documents-from-one-collection-to-another-collection
First 3500 into train, the next 558 into test, as the _id goes from 1 to 4058.  Use $gt: 3500 and $lt: 5301.
It would also be simple (simpler perhaps) to modify the carstens_into_mongo function below.
"""


BATCH_SIZE = {
    'snli': {'train': 68,
             'dev': 50,
             'test': 50},
    'carstens': {'all': 101,
                 'train': 100,
                 'test': 558}
}
BUFFER_FACTORS = {
    'snli': {'train': 4,
             'dev': 4,
             'test': 4},
    'carstens': {'all': 4,
                 'train': 35,
                 'test': 1}
}
COLLECTION_SIZE = {
    'snli': {'train': 55012,
             'dev': 10000,
             'test': 10000},
    'carstens': {'all': 4058,
                 'train': 3500,
                 'test': 558}
}
ENCODING_TO_LABEL = {0: 'neutral',
                     1: 'entailment',
                     2: 'contradiction'}
LABEL_TO_ENCODING = {'neutral': 0,
                     'entailment': 1,
                     'contradiction': 2,
                     '-': 0}  # will have to deal with this properly!!!
LONGEST_SENTENCE_SNLI = 402
NUM_ITERS = {
    'snli': {'train': 809,
             'dev': 200,
             'test': 200},
    'carstens': {'all': 40,
                 'train': 35,
                 'test': 1}
}
NUM_LABELS = 3
REPORT_EVERY = {
    'snli': {'train': 101,
             'dev': 20,
             'test': 20},
    'carstens': {'all': 4,
                 'train': 5,
                 'test': 1
                 }
}


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
    encoding[0, LABEL_TO_ENCODING[label]] = 1
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


class RandomizedGenerator:
    def __init__(self, db='snli', collection='train'):
        self._db_name = db
        self._collection = collection
        self._batch_size = BATCH_SIZE[db][collection]
        self._buffer_factor = BUFFER_FACTORS[db][collection]
        self._db = Carstens() if db == 'carstens' else SNLIDb()
        self._gen = self._db.repository(self._collection).find_all()
        self._i_yielded = 0
        self._i_buffered = 0
        self._buffered = []
        self._fill_buffer()

    def _fill_buffer(self):
        for i in range(self._batch_size * self._buffer_factor):
            if self._gen.alive:
                self._buffered.append(next(self._gen))
                self._i_buffered += 1
            else:
                break  # don't waste time iterating further if we're at the end

    def next(self):
        if len(self._buffered) == 0:
            self._raise_exception()
        sample = np.random.choice(self._buffered, size=1)[0]
        self._buffered.remove(sample)
        if len(self._buffered) == 0:
            if self._gen.alive:
                self._fill_buffer()
        self._i_yielded += 1
        return sample

    def _raise_exception(self):
        info = 'Attempted to fetch but buffer is empty.  State: ' \
               '%s yielded; %s buffered.' % (self._i_yielded, self._i_buffered)
        info += '\ndb: %s; collection: %s' % (self._db_name, self._collection)
        raise Exception(info)


def get_batch_gen(db, collection):
    # each batch is a float64 array of shape: BATCH_SIZE x None x WORD_EMBED_DIM
    # but the labels is just batch_size x 3
    gen = RandomizedGenerator(db, collection)
    while True:
        premises = []
        hypotheses = []
        labels = []
        for i in range(BATCH_SIZE[db][collection]):
            doc = gen.next()
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


def pad_out_sentences(batch,
                      pad_max=False,
                      premise_pad_length=LONGEST_SENTENCE_SNLI,
                      hypothesis_pad_length=LONGEST_SENTENCE_SNLI):
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
    return sents, matrices


def count_no_gold_labels(collection):
    db = SNLIDb()
    train_all = db.repository(collection).find_all()
    count = 0
    all = 0
    for doc in train_all:
        if doc['gold_label'] == '-':
            count += 1
        all += 1
    print('Count = %s' % count)
    print('Percentage = %s' % (count / all))
    # train: 785, 0.0014


def count_missing_word_vectors(collection):
    nlp = spacy.load('en')
    zero_vector = np.zeros((300,), dtype='float')
    db = SNLIDb()
    all_docs = db.repository(collection).find_all()
    count = 0
    all = 0
    for doc in all_docs:
        for word_vector in string_to_array(doc['premise']).tolist():
            all += 1
            if np.array_equal(word_vector, zero_vector):
                count += 1
        for word_vector in string_to_array(doc['hypothesis']).tolist():
            all += 1
            if np.array_equal(word_vector, zero_vector):
                count += 1
    print('Count = %s' % count)
    print('Percentage = %s' % (count / all))
    # train:
    # dev: 182, 0.00077
    # test:


def carstens_into_mongo(file_path='/home/hanshan/carstens.csv'):
    X = pd.read_csv(file_path, header=None)
    label = {
        'n': 'neutral',
        's': 'entailment',
        'a': 'contradiction'
    }
    db = Carstens()
    nlp = spacy.load('en')
    id = 0
    for x in X.iterrows():
        id += 1
        doc = {
            '_id': id,
            'sentence1': x[1][3],
            'sentence2': x[1][4],
            'gold_label': label[x[1][5]]
        }
        doc['premise'] = array_to_string(sentence_matrix(doc['sentence1'], nlp))
        doc['hypothesis'] = array_to_string(sentence_matrix(doc['sentence2'], nlp))
        db.all.insert_one(doc)
    raise Exception('Could do the train and test split here, too')


if __name__ == '__main__':
    carstens_into_mongo()
