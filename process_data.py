import mongoi
import spacy
import numpy as np
import util
import labeling


"""
Process from start to finish:
* Make sure the OOV_VECTORS dictionary has the entry for the db
1. Import data into mongo, e.g.:
    mongoimport --db snli --collection train
    "/home/hanshan/PycharmProjects/argmin/data/....jsonl"
2. find_oov()
3. remove_no_gold_label_samples()
4. generate_friendly_ids()
    - this will be an extra field this time
5. generate_label_encodings()
6. generate_sentence_matrices()
    - turn sentences into spacy docs
    - grab GloVe vectors from spacy into matrices
    - put in place OOV random vectors
    - prepend NULLs
"""


""" Functions for pre-processing data in mongo """


def find_oov(db):
    print('Finding OOV words '
          'and creating random vector dictionary '
          'for db "%s"' % db)
    oov_words = []
    oov_vectors = {}
    nlp = spacy.load('en')
    zero_vector = np.zeros(300,)
    for collection in mongoi.COLLECTIONS[db]:
        count = 0
        repository = mongoi.get_repository(db, collection)
        for doc in repository.find_all():
            got_one = False
            premise_doc = nlp(doc['sentence1'])
            for token in premise_doc:
                if np.array_equal(token.vector, zero_vector):
                    oov_words.append(token.text)
                    got_one = True
            hypothesis_doc = nlp(doc['sentence2'])
            for token in hypothesis_doc:
                if np.array_equal(token.vector, zero_vector):
                    oov_words.append(token.text)
                    got_one = True
            if got_one:
                count += 1
        print('%s samples with OOV in %s.%s' % (count, db, collection))
    for word in set(oov_words):
        oov_vectors[word] = np.random.rand(1, 300)
    util.save_pickle(oov_vectors, 'oov_vectors_%s.pkl' % db)


def remove_no_gold_label_samples(db):
    print('Removing no gold label samples from %s...' % db)
    for collection in mongoi.COLLECTIONS[db]:
        print('Deleting no gold labels from collection: "%s"' % collection)
        deleted_count = 0
        repository = mongoi.get_repository(db, collection)
        for doc in repository.find_all():
            if doc['gold_label'] == '-':
                repository.delete_one(doc['_id'])
                deleted_count += 1
        print('Deleted %s documents' % deleted_count)
        print('Remaining count in collection: %s' % repository.count())
    print('Finished successfully.')


def generate_friendly_ids(db):
    print('Generating friendly ids for %s...' % db)
    id = 1
    for collection in mongoi.COLLECTIONS[db]:
        print('Working on collection: %s' % collection)
        repository = mongoi.get_repository(db, collection)
        for doc in repository.find_all():
            repository.update_one(doc['_id'], {'id': id})
            id += 1
    print('Completed successfully.')


def generate_label_encodings(db):
    print('Generating label encodings for %s...' % db)
    for collection in mongoi.COLLECTIONS[db]:
        print('Working on collection: %s' % collection)
        repository = mongoi.get_repository(db, collection)
        for doc in repository.find_all():
            encoding = encode(doc['gold_label'])
            repository.update_one(doc['_id'],
                                  {'label_encoding':
                                       mongoi.array_to_string(encoding)})
    print('Completed successfully.')


def encode(label):
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


""" Functions for finding statistics """


def find_max_length(db):
    longest_overall = 0
    nlp = spacy.load('en')
    for collection in mongoi.COLLECTIONS[db]:
        longest_in_collection = 0
        repository = mongoi.get_repository(db, collection)
        for doc in repository.find_all():
            spacy_premise = nlp(doc['sentence1'])
            if len(spacy_premise) > longest_in_collection:
                longest_in_collection = len(spacy_premise)
            spacy_hypothesis = nlp(doc['sentence2'])
            if len(spacy_hypothesis) > longest_in_collection:
                longest_in_collection = len(spacy_hypothesis)
        if longest_in_collection > longest_overall:
            longest_overall = longest_in_collection


if __name__ == '__main__':
    find_oov('carstens')
    #remove_no_gold_label_samples('mnli')
    #generate_friendly_ids('mnli')
    #generate_label_encodings('mnli')
    generate_sentence_matrices('carstens')
