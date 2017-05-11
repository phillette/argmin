"""For pre-processing data."""
import mongoi
import spacy
import numpy as np
import util
import labeling
import errors
import oov


"""
Process from start to finish:
1. Import data into mongo, e.g.:
    mongoimport --db snli --collection train
    "/home/hanshan/PycharmProjects/argmin/data/....jsonl"
2. generate_friendly_ids()
3. remove_no_gold_label_samples()
4. generate_friendly_ids()
    - this will be an extra field this time
5. generate or get oov
6. decide how to handle oov
7. generate_sentence_matrices()
    - turn sentences into spacy docs
    - grab GloVe vectors from spacy into matrices
    - put in place OOV random vectors
    - prepend NULLs
Wrap this entire process in a function called pre_process(db_name).
"""


"""
NOTE:
There is terrible inefficiency in looping over the entire collection
as many times as there are things to do to the samples. I should fix
this by rewriting the functions to operate on one sample, and then
perhaps have another function that iterates and applies one, or a list
of functions to each sample, in order. Then, to wrap the whole process
I could call the looping function with an ordered list of operation
functions to apply to each sample.
"""


def remove_no_gold_label_samples(db_name):
    """Removes samples without gold labels.

    This is specifically for SNLI, where a number of samples had
    no gold label, instead just a '-'. We do not want to train with
    these samples so we remove them.

    Question: does MNLI have them, too? I've kept the db_name argument
    in any case.

    Args:
      db_name: the name of the database (dataset) to work on.
    Raises:
      DbNotFoundError: if the db_name is not in mongoi.COLLECTIONS.keys(),
        which is intended to provide the list of collections to operate
        on.
    """
    if db_name not in mongoi.COLLECTIONS.keys():
        raise errors.DbNotFoundError(db_name)
    print('Removing no gold label samples from %s...' % db_name)

    for collection_name in mongoi.COLLECTIONS[db_name]:
        print('Deleting no gold labels from collection: "%s"' % collection_name)
        deleted_count = 0
        repository = mongoi.get_repository(db_name, collection_name)
        for sample in repository.find_all():
            if sample['gold_label'] == '-':
                repository.delete(sample)
                deleted_count += 1
        print('Deleted %s documents' % deleted_count)
        print('Remaining count in collection: %s' % repository.count())

    print('Finished successfully.')


def generate_friendly_ids(db_name):
    """Create id attributes_name for all samples in the db.

    The are considered friendly since they are a simple int,
    which are easier to work with that the long and complicated
    _id attributes that mongo creates, and which come with the
    mongo_import of the huge data in SNLI and MNLI. Creating
    these does away with the need to delete and reinsert every
    record if we wished to update the _id attribute.

    Args:
      db_name: the name of the database to operate on.
    Raises:
      DbNotFoundError: if the db_name is not in mongoi.COLLECTIONS.keys(),
        which is intended to provide the list of collections to operate
        on.
    """
    if db_name not in mongoi.COLLECTIONS.keys():
        raise errors.DbNotFoundError(db_name)
    print('Generating friendly ids for %s...' % db_name)

    id = 1
    for collection_name in mongoi.COLLECTIONS[db_name]:
        print('Working on collection: %s' % collection_name)
        repository = mongoi.get_repository(db_name, collection_name)
        for sample in repository.find_all():
            sample['id'] = id
            repository.update(sample)
            id += 1

    print('Completed successfully.')


def generate_label_encodings(db_name):
    """Attach a label_encoding to each sample.

    The encoding is a numpy array representing the one-hot vector
    encoding of a sample's label. Each encoding is a row vector of
    size three.

    Args:
      db_name: the name of the db to operate on.
    Raises:
      DbNotFoundError: if the db_name is not in mongoi.COLLECTIONS.keys(),
        which is intended to provide the list of collections to operate
        on.
    """
    if db_name not in mongoi.COLLECTIONS.keys():
        raise errors.DbNotFoundError(db_name)
    print('Generating label encodings for %s...' % db_name)

    for collection_name in mongoi.COLLECTIONS[db_name]:
        print('Working on collection: %s' % collection_name)
        repository = mongoi.get_repository(db_name, collection_name)
        for sample in repository.all():
            encoding = encode(sample['gold_label'])
            sample['label_encoding'] = mongoi.array_to_string(encoding)
            repository.update(sample)

    print('Completed successfully.')


def generate_sparse_encodings(db_name):
    """Attach a sparse_encoding to each sample.

    Sparse label encodings are to be plugged into
    tf.nn.sparse_softmax_cross_entropy_with_logits
    and are therefore to be preferred over the one-hot
    encoding. The one-hot encoding I intend to remove.

    Args:
      db_name: the name of the db to operate on.
    Raises:
      DbNotFoundError: if the db_name is not in mongoi.COLLECTIONS.keys(),
        which is intended to provide the list of collections to operate
        on.
    """
    if db_name not in mongoi.COLLECTIONS.keys():
        raise errors.DbNotFoundError(db_name)
    print('Generating sparse labels for %s...' % db_name)

    for collection_name in mongoi.COLLECTIONS[db_name]:
        print('Working on collection: %s' % collection_name)
        repository = mongoi.get_repository(db_name, collection_name)
        for sample in repository.all():
            label = sample['gold_label']
            sparse_encoding = labeling.LABEL_TO_ENCODING[label]
            sample['sparse_encoding'] = sparse_encoding
            repository.update(sample)

    print('Completed successfully.')


def encode(label):
    """Encode a label to a y-vector.

    Args:
      label: the text of the label to encode.
    Returns:
      numpy array representing one-hot vector of shape (1, 3)
        which encodes the label.
    Raises:
      LabelNotFoundError: if the label is not found in the
        labeling.LABEL_TO_ENCODING dictionary. This may happen
        if a new data set is loaded and the particular labels
        they use have not been added to the dictionary.
    """
    if label not in labeling.LABEL_TO_ENCODING.keys():
        raise errors.LabelNotFoundError(label)

    encoding = np.zeros((1, 3), dtype='float64')
    encoding[0, labeling.LABEL_TO_ENCODING[label]] = 1

    return encoding


def generate_sentence_matrices(db):
    print('Generating sentence matrices for %s...' % db)
    nlp = spacy.load('en')
    null_vector = util.load_pickle('NULL_glove_vector.pkl')
    oov_vectors = load_oov_vectors(db)
    for collection in mongoi.COLLECTIONS[db]:
        print('Working on collection: %s' % collection)
        repository = mongoi.get_repository(db, collection)
        for doc in repository.find_all():
            premise = sentence_matrix(doc['sentence1'],
                                      nlp,
                                      null_vector,
                                      oov_vectors)
            hypothesis = sentence_matrix(doc['sentence2'],
                                         nlp,
                                         null_vector,
                                         oov_vectors)
            repository.update_one(doc['_id'],
                                  {'premise':
                                       mongoi.array_to_string(premise),
                                   'hypothesis':
                                       mongoi.array_to_string(hypothesis)})
    print('Completed successfully.')


def sentence_matrix(sentence, nlp, null_vector, oov_vectors):
    doc = nlp(sentence)
    matrix = np.vstack(
        list(
            get_vector(t, oov_vectors).reshape((1, 300)) for t in doc
        ))
    matrix = np.vstack([null_vector, matrix])
    return matrix


def get_vector(token, oov_vectors):
    if token.text in oov_vectors.keys():
        return oov_vectors[token.text]
    else:
        return token.vector


def load_oov_vectors(db):
    return util.load_pickle('oov_vectors_%s.pkl' % db)
