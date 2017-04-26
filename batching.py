import mongoi
import numpy as np
import util
import stats
import labeling


NULL_VECTOR = util.load_pickle('NULL_glove_vector.pkl')
PREFERRED_BATCH_SIZES = {
    'snli': {
        'train': 4,
        'dev': 4,
        'test': 4
    },
    'carstens': {

    }
}


class Batch:
    def __init__(self, ids, premises, hypotheses, labels):
        self.ids = ids
        self.premises = premises
        self.hypotheses = hypotheses
        self.labels = labels


class RandomGenerator:
    def __init__(self, db_name='snli', collection='train', buffer_size=100):
        self._db_name = db_name
        self._collection = collection
        self._repository = mongoi.get_repository(db_name, collection)
        self._gen = self._repository.batch()
        self._buffer_size = buffer_size
        self._db_yielded = 0
        self._i_yielded = 0
        self._buffer = []
        self._fill_buffer()

    def _fill_buffer(self):  # I wish I could do this async!
        while len(self._buffer) < self._buffer_size:
            if self._gen.alive:
                next_doc = next(self._gen)
                self._db_yielded += 1
                self._buffer.append(next_doc)
            else:
                break  # don't waste time iterating further if we're at the end

    def alive(self):
        return self._gen.alive or len(self._buffer) > 0

    def next(self):
        if len(self._buffer) == 0:
            self._raise_exception()
        sample = np.random.choice(self._buffer, size=1)[0]
        self._buffer.remove(sample)
        if len(self._buffer) == 0:
            self._fill_buffer()
        self._i_yielded += 1
        return sample

    def _raise_exception(self):
        info = 'Attempted to fetch but buffer is empty.'
        info += '\nState: %s yielded; %s buffered; %s db yielded.' % (self._i_yielded,
                                                                      len(self._buffer),
                                                                      self._db_yielded)
        info += '\ndb: %s; collection: %s' % (self._db_name,
                                              self._collection)
        raise Exception(info)


def get_batch_gen(db, collection, batch_size=None):
    # if a batch_size is not selected, default to the preferred
    if not batch_size:
        batch_size = PREFERRED_BATCH_SIZES[db][collection]
    # the random generator to use
    gen = RandomGenerator(db, collection)
    while True:
        ids = []
        premises = []
        hypotheses = []
        labels = []
        for _ in range(batch_size):
            if gen.alive():  # if there are no more records, don't try and find one
                doc = gen.next()
                id = doc['_id']
                premise = mongoi.string_to_array(doc['premise'])
                hypothesis = mongoi.string_to_array(doc['hypothesis'])
                premise = prepend_null(premise)        # comment this out to go back to normal
                hypothesis = prepend_null(hypothesis)  # comment this out to go back to normal
                label = encode(doc['gold_label'])
                ids.append(id)
                premises.append(premise)
                hypotheses.append(hypothesis)
                labels.append(label)
        batch = Batch(ids, premises, hypotheses, labels)
        pad_sentences(batch)
        add_third_dimensions(batch)
        yield batch


def encode(label):
    encoding = np.zeros((1, 3), dtype='float64')
    encoding[0, labeling.LABEL_TO_ENCODING[label]] = 1
    return encoding


def pad_sentences(batch,
                  pad_max=False,
                  pad_length=stats.LONGEST_SENTENCE_SNLI):
    # at this stage the premises and hypotheses are still lists of matrices
    if pad_max:
        premise_pad_length = max([premise.shape[0] for premise in batch.premises])
        hypothesis_pad_length = max([hypothesis.shape[0] for hypothesis in batch.hypotheses])
    batch.premises = list(pad_sentence(premise, pad_length) for premise in batch.premises)
    batch.hypotheses = list(pad_sentence(hypothesis, pad_length) for hypothesis in batch.hypotheses)


def pad_sentence(sentence, desired_length):
    original_length = sentence.shape[0]
    word_embed_dim = sentence.shape[1]
    if original_length < desired_length:
        sentence = np.concatenate([sentence,
                                   np.zeros((desired_length - original_length,
                                             word_embed_dim))],
                                  axis=0)
    return sentence


def prepend_null(sentence_matrix):
    return np.vstack([NULL_VECTOR, sentence_matrix])


def add_third_dimensions(batch):
    batch.premises = np.concatenate(list(M[np.newaxis, ...] for M in batch.premises), axis=0)
    batch.hypotheses = np.concatenate(list(M[np.newaxis, ...] for M in batch.hypotheses), axis=0)
    batch.labels = np.vstack(batch.labels)


def num_iters(db, collection, batch_size):
    collection_size = stats.COLLECTION_SIZE[db][collection]
    return np.ceil(collection_size / batch_size)
