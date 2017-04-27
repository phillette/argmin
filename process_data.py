import mongoi
import spacy
import numpy as np
import pandas as pd
import util
import labeling


"""
Process from start to finish:
* Make sure the OOV_VECTORS dictionary has the entry for the db
1. Import data into mongo, e.g.:
    mongoimport --db snli --collection train
    "/home/hanshan/PycharmProjects/argmin/data/....jsonl"
2. remove_no_gold_label_samples()
3. generate_friendly_ids()
    - this will be an extra field this time
4. generate_label_encodings()
5. generate_sentence_matrices()
    - turn sentences into spacy docs
    - grab GloVe vectors from spacy into matrices
    - put in place OOV random vectors
    - prepend NULLs
"""


NULL_VECTOR = util.load_pickle('NULL_glove_vector.pkl')
OOV_VECTORS = {
    'snli': util.load_pickle('oov_vectors_snli.pkl'),
    #'mnli': util.load_pickle('oov_vectors_mnli.pkl')
}


""" Functions for pre-processing data in mongo """


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
    for collection in mongoi.COLLECTIONS[db]:
        print('Working on collection: %s' % collection)
        repository = mongoi.get_repository(db, collection)
        for doc in repository.find_all():
            premise = sentence_matrix(doc['sentence1'], nlp, db)
            hypothesis = sentence_matrix(doc['sentence2'], nlp, db)
            repository.update_one(doc['_id'],
                                  {'premise':
                                       mongoi.array_to_string(premise),
                                   'hypothesis':
                                       mongoi.array_to_string(hypothesis)})
    print('Completed successfully.')


def sentence_matrix(sentence, nlp, db):
    doc = nlp(sentence)
    matrix = np.vstack(list(get_vector(t, db).reshape((1, 300)) for t in doc))
    return prepend_null(matrix)


def get_vector(token, db):
    if token.text in OOV_VECTORS.keys():
        return OOV_VECTORS[db][token.text]
    else:
        return token.vector


def prepend_null(sentence_matrix):
    return np.vstack([NULL_VECTOR, sentence_matrix])


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



""" Functions for dealing with OOV """


def _generate_oov_vectors():
    mv = util.load_pickle('missing_vectors.pkl')
    oov_vectors = {}  # word: vector
    all_words = set(list(mv['train']['words'].values())
                    + list(mv['dev']['words'].values())
                    + list(mv['test']['words'].values()))
    for word in all_words:
        oov_vectors[word] = np.random.rand(1, 300)
    util.save_pickle(oov_vectors, 'oov_vectors.pkl')


def update_oov_vectors(collections=mongoi.COLLECTIONS['snli']):
    nlp = spacy.load('en')
    missing_vectors = util.load_pickle('missing_vectors.pkl')
    oov_vectors = util.load_pickle('oov_vectors.pkl')
    for collection in collections:
        _update_oov_vectors_per_collection(collection, nlp, missing_vectors, oov_vectors)


def _update_oov_vectors_per_collection(collection, nlp, missing_vectors, oov_vectors):
    no_gold_labels = util.load_pickle('no_gold_label_ids.pkl')
    repository = mongoi.get_repository('snli', collection)
    for doc_id, word in missing_vectors[collection]['words'].items():
        if doc_id not in no_gold_labels[collection]:
            doc = next(repository.get(doc_id))
            _update_oov_vectors_per_document(doc, word, oov_vectors[word], nlp, repository)


def _update_oov_vectors_per_document(doc, word, word_vector, nlp, repository):
    premise_doc = nlp(doc['sentence1'])
    hypothesis_doc = nlp(doc['sentence2'])
    if word in [t.text for t in premise_doc]:
        index = next((t.i for t in premise_doc if t.text == word))
        premise_matrix = mongoi.string_to_array(doc['premise'])
        premise_matrix[index, :] = word_vector
        repository.update_one(doc['_id'], {'premise': mongoi.array_to_string(premise_matrix)})
    if word in [t.text for t in hypothesis_doc]:
        index = next((t.i for t in hypothesis_doc if t.text == word))
        premise_matrix = mongoi.string_to_array(doc['hypothesis'])
        premise_matrix[index, :] = word_vector
        repository.update_one(doc['_id'], {'hypothesis': mongoi.array_to_string(premise_matrix)})


if __name__ == '__main__':
    remove_no_gold_label_samples('mnli')
    generate_friendly_ids('mnli')
    generate_label_encodings('mnli')
