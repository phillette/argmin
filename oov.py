"""Finding and managing out-of-vocabulary tokens."""
import mongoi
import spacy
import errors
import numpy as np
import util


"""
NOTE: would like to have a global oov so that crossing datasets
      that have the same oov can share the vectors for consistency.
      I don't know how often that would be encountered.
      But it may also be better for memory management if individual
      oov vectors were stored in separate pickles and loaded on
      the fly as necessary.
"""


class OOV:
    """Wraps all OOV information for a given dataset.

    Attributes:
      db_name: the name of the database that holds the dataset.
      tokens_to_ids: dictionary of structure oov[token] = token_id.
      ids_to_tokens: dictionary of structure oov[token_id] = token.
      ids_to_random_vectors: dictionary of structure
        oov[token_id] = randomly_initialized_vector.
      samples: dictionary of structure
        oov[collection_name][sample_id] = list(token_ids).
      token_counts: dictionary of structure
        oov[collection_name] = number_of_oov_tokens.
      sample_counts: dictionary of structure
        oov[collection_name] = number_of_oov_samples.
    """

    def __init__(self, db_name):
        """Create a new OOV object."""
        self.db_name = db_name
        self.tokens_to_ids = {}
        self.ids_to_tokens = {}
        self.ids_to_random_vectors = {}
        self.samples = {}
        self.token_counts = {}
        self.sample_counts = {}
        # initialize the dictionaries for each collection now
        self.token_counts['all'] = 0
        self.sample_counts['all'] = 0
        for collection_name in mongoi.COLLECTIONS[db_name]:
            self.samples[collection_name] = {}
            self.token_counts[collection_name] = 0
            self.sample_counts[collection_name] = 0
        # record of the last supplied id so the next one is unique
        self._last_id = 0

    def is_oov(self, token_text):
        """Determine if a token is oov.

        Args:
          token_text: the text of the token.
        Returns:
          True if the token is found in the dictionary, else False.
        """
        return token_text in self.tokens_to_ids.keys()

    def generate_random_vectors(self, size, orientation):
        """Generates random vectors for each oov token.

        Vectors are randomly initialized according to the size and
        orientation provided in the args.

        Args:
          size: the dimensionality of the random vectors to generate.
          orientation: 'row' or 'column' for the orientation of the
            generated vectors.
        Raises:
          UnexpectedValueError: if the orientation is neither 'row'
            nor 'column'.
        """
        for token in self.tokens_to_ids.keys():
            if orientation == 'row':
                random_vector = np.random.rand(1, size)
            elif orientation == 'column':
                random_vector = np.random.rand(size, 1)
            else:
                raise errors.UnexpectedValueError('orientation', orientation)
            self.ids_to_random_vectors[self.tokens_to_ids[token]] \
                = random_vector

    def _new_token_id(self):
        # generates a new unique token id
        new_id = self._last_id + 1
        self._last_id += 1
        return new_id

    def print_counts(self):
        """Prints all count info to the terminal."""
        print('OOV counts for db %s as follows...' % self.db_name)
        print('Overall %s tokens are OOV' % self.token_counts['all'])
        print('Overall %s samples are OOV' % self.sample_counts['all'])
        print('Collection breakdown as follows...')
        for collection_name in mongoi.COLLECTIONS[self.db_name]:
            print('%s tokens are OOV in %s' %
                  (self.token_counts[collection_name], collection_name))
            print('%s samples are OOV in %s' %
                  (self.sample_counts[collection_name], collection_name))

    def report_token(self, token_text, collection_name, sample_id):
        """Adds the token and sample to the data.

        Args:
          token_text: the text (string) of the oov token.
          collection_name: the name of the collection the sample
            comes from.
          sample_id: the friendly id of the sample from which the
            token was taken.
        """
        # we need to check if it has been reported or not to avoid
        # duplicating entries.
        if token_text not in self.tokens_to_ids.keys():
            new_id = self._new_token_id()
            self.tokens_to_ids[token_text] = new_id
            self.ids_to_tokens[new_id] = token_text
            self.token_counts['all'] += 1
            self.token_counts[collection_name] += 1
        # if we have not seen this sample before we need to create
        # a list to which to append the current token.
        if sample_id not in self.samples[collection_name].keys():
            self.samples[collection_name][sample_id] = []
            self.sample_counts['all'] += 1
            self.sample_counts[collection_name] += 1
        self.samples[collection_name][sample_id]\
            .append(self.tokens_to_ids[token_text])

    def save(self):
        """Save this object as a pickle."""
        util.save_pickle(self, _oov_file_path(self.db_name))


def generate_oov(db_name):
    """Determine OOV tokens and samples in a dataset.

    Looks through all collections in a mongo database (that represents
    a dataset), determines which tokens are out-of-vocabulary for the
    GloVe vectors provided by SpaCy, and returns them in an OOV object
    that wraps all information and statistics that can be conveniently
    used for further processing.

    Make sure generate_friendly_ids has been done first because
    this function will use doc['id'] as a key in the dictionary to
    keep track of which samples have OOV.

    Args:
      db_name: the name of the mongo database that holds the dataset.
    Returns:
      An OOV object.
    Raises:
      DbNotFoundError: if the name of the database is not found
        in they key of mongoi.COLLECTIONS.
    """
    if db_name not in mongoi.COLLECTIONS.keys():
        raise errors.DbNotFoundError(db_name)

    print('Finding OOV tokens for db %s' % db_name)
    oov = OOV(db_name)
    nlp = spacy.load('en')
    zero_vector = np.zeros(300,)

    for collection_name in mongoi.COLLECTIONS[db_name]:
        print('Working on collection %s...' % collection_name)
        repository = mongoi.get_repository(db_name, collection_name)
        for sample in repository.find_all():
            premise = nlp(sample['sentence1'])
            hypothesis = nlp(sample['sentence2'])
            for token in premise:
                if np.array_equal(token.vector, zero_vector):
                    oov.report_token(token.text,
                                     collection_name,
                                     sample['id'])
            for token in hypothesis:
                if np.array_equal(token.vector, zero_vector):
                    oov.report_token(token.text,
                                     collection_name,
                                     sample['id'])

    oov.save()
    oov.print_counts()
    print('Completed successfully.')
    return oov


def load(db_name):
    """Loads the pickled OOV object.

    Args:
      db_name: the name of the db for which to load the OOV data.
    Returns:
      An OOV object.
    Raises:
      OOVNotGeneratedError: if the OOV object cannot be found for
        the given db_name. This would indicate generate_oov had
        not yet been run.
    """
    return util.load_pickle(_oov_file_path(db_name))


def _oov_file_path(db_name):
    return 'oov_%s.pkl' % db_name
