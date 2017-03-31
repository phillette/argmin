from mongoi import CarstensDb, SNLIDb, string_to_array
import numpy as np


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
                     '-': 0}  # this is an issue
LONGEST_SENTENCE_SNLI = 402
MISSING_WORD_VECTOR_COUNTS = {  # don't double count with no gold labels!
    'train': 0,
    'dev': 0,
    'test': 0
}
NO_GOLD_LABEL_COUNTS = {
    'train': 0,
    'dev': 0,
    'test': 0
}
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


def add_third_dimensions(batch):
    batch.premises = np.concatenate(list(M[np.newaxis, ...] for M in batch.premises), axis=0)
    batch.hypotheses = np.concatenate(list(M[np.newaxis, ...] for M in batch.hypotheses), axis=0)
    batch.labels = np.vstack(batch.labels)


class Batch:
    def __init__(self, premises, hypotheses, labels):
        self.premises = premises
        self.hypotheses = hypotheses
        self.labels = labels


def encode(label):
    encoding = np.zeros((1, 3), dtype='float64')
    encoding[0, LABEL_TO_ENCODING[label]] = 1
    return encoding


def get_batch_gen(db, collection, type='gen'):
    if type == 'gen':
        gen = RandomizedGeneratorFromGen(db, collection)
    elif type == 'id':
        gen = RandomizedGeneratorFromIDs(db, collection)
    else:
        raise Exception('Unexpected generator type: %s' % type)
    while True:
        premises = []
        hypotheses = []
        labels = []
        for _ in range(BATCH_SIZE[db][collection]):
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


def pad_sentence(sentence, desired_length):
    original_length = sentence.shape[0]
    word_embed_dim = sentence.shape[1]
    if original_length < desired_length:
        sentence = np.concatenate([sentence,
                                   np.zeros((desired_length - original_length,
                                             word_embed_dim))],
                                  axis=0)
    return sentence


class RandomizedGeneratorFromGen:
    def __init__(self, db='snli', collection='train'):
        self._db_name = db
        self._collection = collection
        self._batch_size = BATCH_SIZE[db][collection]
        self._buffer_factor = BUFFER_FACTORS[db][collection]
        self._db = CarstensDb() if db == 'carstens' else SNLIDb()
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


class RandomizedGeneratorFromIDs:
    def __init__(self, db, collection):
        self._collection = collection
        self._batch_size = BATCH_SIZE[db][collection]
        self._db = CarstensDb() if db == 'carstens' else SNLIDb()
        self._ids = list(range(COLLECTION_SIZE[db][collection]))

    def next(self):
        new_id = np.random.choice(a=self._ids,
                                  size=1)[0]
        self._ids.remove(new_id)
        return self._db.repository(self._collection).find('_id', new_id)  # will this be slow?
