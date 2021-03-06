import mongoi
import numpy as np
import stats
import errors
import tensorflow_fold as td


PREFERRED_BATCH_SIZES = {
    'snli': {
        'train': 32,
        'dev': 32,
        'test': 32
    },
    'carstens': {
        'all': 32
    }
}


class Batcher:
    """Base class for a wrapper for a batch generator.

    This class is intended to be reusable across epochs. It therefore in theory
    accepts a generator of generators. In practice we want to wrap a function
    that retrieves a generator from MongoDB. What is done with that are details
    left entirely up to the conscience of individual subclasses.

    At the other end, the main function that needs to be implemented is
    next_batch().

    Attributes:
      batch_size: Integer, pre-defined.
      data_generator: Function that returns pymongo.cursor.Cursor.
    """

    def __init__(self, batch_size, data_generator):
        """Create a new Batcher.

        Args:
          batch_size: Integer, the number of samples per batch.
          data_generator: Function that returns pymongo.cursor.Cursor.
        """
        self.batch_size = batch_size
        self.data_generator = data_generator

    def next_batch(self):
        raise NotImplementedError()


class FoldBatcher(Batcher):
    """Batcher for preparing data for TensorFlow Fold models.

    We need to:
    - call the data_generator() function and keep a list of the data.
    -
    """

    def __init__(self, batch_size, data_generator, compiler):
        """Create a new FoldBatcher.

        Args:
          batch_size: Integer, the number of samples per batch.
          data_generator: Function that returns pymongo.cursor.Cursor.
          compiler: td.Compiler, from a loaded model.
        """
        super(FoldBatcher, self).__init__(batch_size, data_generator)
        self.compiler = compiler
        self.data = compiler.build_loom_inputs(data_generator())

    def next_batch(self):
        pass


class Batch:
    """Wrapper for batch data."""
    def __init__(self, ids, premises, hypotheses, labels):
        """Create a new Batch object.

        Args:
          ids: list of friendly ids for the samples in the batch
          premises: list of premise matrices
          hypotheses: list of hypothesis matrices
          labels: list of label encodings
        """
        self.ids = ids
        self.premises = add_third_dimension(premises)
        self.hypotheses = add_third_dimension(hypotheses)
        self.labels = np.vstack(labels)

    def details(self):
        return 'Ids: %s\n' \
               'Premises shape: %s\n' \
               'Hypotheses shape: %s\n' \
               'Labels shape: %s' \
               % (' '.join(str(id) for id in self.ids),
                  self.premises.shape,
                  self.hypotheses.shape,
                  self.labels.shape)


class RandomGenerator:
    def __init__(self, db_name='snli',
                 collection='train',
                 buffer_size=100,
                 transfer=False):
        self._db_name = db_name
        self._collection = collection
        self._repository = mongoi.get_repository(db_name, collection)
        self._gen = self._repository.batch(transfer)
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
        info += '\nState: %s yielded; %s buffered; %s db yielded.' \
                % (self._i_yielded,
                   len(self._buffer),
                   self._db_yielded)
        info += '\ndb: %s; collection: %s' % (self._db_name,
                                              self._collection)
        raise Exception(info)


class TransferBatch:
    def __init__(self, ids, X, labels):
        self.ids = ids
        self.X = np.vstack(X)
        self.labels = np.vstack(labels)


def add_third_dimension(sentence_matrices):
    """Adds third 'batch' dimension to batch data.

    Args:
      sentence_matrices: a list of sentence matrices.
    Raises:
      EmptyListError: if the list of sentence matrices is empty.
    """
    if len(sentence_matrices) == 0:
        raise errors.EmptyListError('sentence_matrices')
    return np.concatenate(list(M[np.newaxis, ...]
                               for M in sentence_matrices),
                          axis=0)


def get_batch_gen(db, collection, batch_size=None):
    if not batch_size:
        batch_size = get_batch_size(db, collection)
    gen = RandomGenerator(db, collection, buffer_size=min(100, batch_size * 10))
    while True:
        ids = []
        premises = []
        hypotheses = []
        labels = []
        pad_length = 0
        for _ in range(batch_size):
            if gen.alive():
                doc = gen.next()
                id = doc['id']
                premise = mongoi.string_to_array(doc['premise'])
                hypothesis = mongoi.string_to_array(doc['hypothesis'])
                label = mongoi.string_to_array(doc['label_encoding'])
                ids.append(id)
                premises.append(premise)
                hypotheses.append(hypothesis)
                labels.append(label)
                pad_length = update_pad_length(pad_length,
                                               premise,
                                               hypothesis)
        premises = pad_sentences(premises, pad_length)
        hypotheses = pad_sentences(hypotheses, pad_length)
        batch = Batch(ids, premises, hypotheses, labels)
        yield batch


def get_batch_gen_transfer(db, collection, batch_size=None):
    # if a batch_size is not selected, default to the preferred
    if not batch_size:
        batch_size = get_batch_size(db, collection)
    gen = RandomGenerator(db,
                          collection,
                          buffer_size=batch_size * 2,
                          transfer=True)
    while True:
        ids = []
        X = []
        labels = []
        for _ in range(batch_size):
            if gen.alive():
                doc = gen.next()
                id = doc['id']
                x = mongoi.string_to_array(doc['features'])
                label = mongoi.string_to_array(doc['label_encoding'])
                ids.append(id)
                X.append(x)
                labels.append(label)
        batch = TransferBatch(ids, X, labels)
        yield batch


def get_batch_size(db, collection):
    collection_size = stats.COLLECTION_SIZE[db][collection]
    return min(np.floor(collection_size / 10), 32)


def num_iters(db, collection, batch_size=None, subset_size=None):
    if not batch_size:
        batch_size = PREFERRED_BATCH_SIZES[db][collection]
    if subset_size:
        return int(np.ceil(subset_size / batch_size))
    collection_size = stats.COLLECTION_SIZE[db][collection]
    return int(np.ceil(collection_size / batch_size))


def pad_sentences(sentences, pad_length):
    """Pad out a list of sentences with zero vectors.

    Args:
      sentences: list of sentence matrices.
      pad_length: integer length to pad out to.
    """
    return list(pad_sentence(premise, pad_length)
                for premise in sentences)


def pad_sentence(sentence, pad_length):
    """Pad out a sentence with zero vectors.

    Args:
      sentence: sentence matrix to pad out.
      pad_length: integer length to pad out to.
    """
    original_length = sentence.shape[0]
    word_embed_dim = sentence.shape[1]
    if original_length < pad_length:
        sentence = np.concatenate(
            [sentence,
             np.zeros((pad_length - original_length,
                       word_embed_dim))],
            axis=0)
    return sentence


def update_pad_length(pad_length, premise, hypothesis):
    premise_length = premise.shape[0]
    if pad_length < premise_length:
        pad_length = premise_length
    hypothesis_length = hypothesis.shape[0]
    if pad_length < hypothesis_length:
        pad_length = hypothesis_length
    return pad_length
