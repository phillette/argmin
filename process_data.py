import mongoi
import spacy
import numpy as np
import itertools
import pandas as pd
from batching import pad_sentence
import util


"""
Note that to split the Carstens data into train and test, I used a variant on this code:
http://stackoverflow.com/questions/27039083/mongodb-move-documents-from-one-collection-to-another-collection
First 3500 into train, the next 558 into test, as the _id goes from 1 to 4058.  Use $gt: 3500 and $lt: 5301.
It would also be simple (simpler perhaps) to modify the carstens_into_mongo function below.
"""


def change_doc_id(doc, new_id, repository):
    repository.delete_one(doc['_id'])
    doc['_id'] = new_id
    repository.insert_one(doc)


def fix_snli_ids():
    db = mongoi.SNLIDb()
    id = 0
    all_train = db.train.find_all()
    all_dev = db.dev.find_all()
    all_test = db.test.find_all()
    for _ in range(db.train.count()):
        id += 1
        doc = next(all_train)
        change_doc_id(doc, id, db.train)
    for _ in range(db.dev.count()):
        id += 1
        doc = next(all_dev)
        change_doc_id(doc, id, db.dev)
    for _ in range(db.test.count()):
        id += 1
        doc = next(all_test)
        change_doc_id(doc, id, db.test)


def sentence_matrix(sentence, nlp):
    doc = nlp(sentence)
    M = np.vstack(list(t.vector.reshape((1, 300)) for t in doc))
    return M


def matrices_into_mongo(nlp, collection):
    db = mongoi.SNLIDb()
    index = 0
    for doc in db.repository(collection).find_all():
        index += 1
        premise = sentence_matrix(doc['sentence1'], nlp)
        hypothesis = sentence_matrix(doc['sentence2'], nlp)
        db.repository(collection).update_one(doc['_id'], {'premise': mongoi.array_to_string(premise),
                                                          'hypothesis': mongoi.array_to_string(hypothesis)})
        print(index)


def find_max_length():
    db = mongoi.SNLIDb()
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


def _get_missing_gold_label_ids(db, collection):
    ids = []
    query = db.repository(collection).find({'gold_label': '-'})
    for doc in query:
        ids.append(doc['_id'])
    return ids


def _get_no_gold_labels():
    """
    Counts of no gold labels as follows.
    train: 785 / 550,012 = 1.4%
    dev:   158 / 10,000  = 1.6%
    test:  176 / 10,000  = 1.8%
    """
    db = mongoi.SNLIDb()
    train_ids = _get_missing_gold_label_ids(db, 'train')
    dev_ids = _get_missing_gold_label_ids(db, 'dev')
    test_ids = _get_missing_gold_label_ids(db, 'test')
    util.save_pickle(train_ids, 'train_no_gold_label_ids.pkl')
    util.save_pickle(dev_ids, 'dev_no_gold_label_ids.pkl')
    util.save_pickle(test_ids, 'test_no_gold_label_ids.pkl')
    return train_ids, dev_ids, test_ids


def _missing_word_vectors_per_collection(db, collection, nlp, zero_vector):
    ids = []
    words = {}
    sentences = {}
    for doc in db.repository(collection).find_all():
        premise_missing, premise_zeros = _missing_word_vectors_per_sentence(doc, 'premise', nlp, zero_vector)
        hypothesis_missing, hypothesis_zeros = _missing_word_vectors_per_sentence(doc, 'hypothesis', nlp, zero_vector)
        if premise_missing or hypothesis_missing:
            ids.append(doc['_id'])
        if premise_missing:
            for word in premise_zeros:
                words[doc['_id']] = word
                sentences[doc['_id']] = doc['sentence1']
        if hypothesis_missing:
            for word in hypothesis_zeros:
                words[doc['_id']] = word
                sentences[doc['_id']] = doc['sentence2']
    return ids, words, sentences


def _missing_word_vectors_per_sentence(doc, array_attr, nlp, zero_vector):
    text_attr = 'sentence1' if array_attr == 'premise' else 'sentence2'
    spacy_doc = nlp(doc[text_attr])
    words = mongoi.string_to_array(doc[array_attr]).tolist()
    zeros = []
    for i in range(len(words)):
        if np.array_equal(words[i], zero_vector):
            zeros.append(spacy_doc[i].text)
    return len(zeros) > 0, zeros


def _get_missing_word_vectors():
    nlp = spacy.load('en')
    zero_vector = np.zeros((300,), dtype='float')
    db = mongoi.SNLIDb()
    train_ids, train_words, train_sents = _missing_word_vectors_per_collection(db, 'train', nlp, zero_vector)
    dev_ids, dev_words, dev_sents = _missing_word_vectors_per_collection(db, 'dev', nlp, zero_vector)
    test_ids, test_words, test_sents = _missing_word_vectors_per_collection(db, 'test', nlp, zero_vector)
    missing_vectors = {
        'train': {
            'ids': train_ids,
            'words': train_words,
            'sents': train_sents
        },
        'dev': {
            'ids': dev_ids,
            'words': dev_words,
            'sents': dev_sents
        },
        'test': {
            'ids': test_ids,
            'words': test_words,
            'sents': test_sents
        }
    }
    util.save_pickle(missing_vectors, 'missing_vectors.pkl')
    return missing_vectors


def missing_word_vectors():
    missing_vectors = util.load_pickle('missing_vectors.pkl')
    return missing_vectors


# IDEA: look at inter-annotator DISAGREEMENT and remove those observations
# (maybe makes the dataset cleaner)?  Worth a test...


def carstens_into_mongo(file_path='/home/hanshan/carstens.csv'):
    # should edit this to do the train-test split (3500-558)
    X = pd.read_csv(file_path, header=None)
    label = {
        'n': 'neutral',
        's': 'entailment',
        'a': 'contradiction'
    }
    db = mongoi.CarstensDb()
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
        doc['premise'] = mongoi.array_to_string(sentence_matrix(doc['sentence1'], nlp))
        doc['hypothesis'] = mongoi.array_to_string(sentence_matrix(doc['sentence2'], nlp))
        db.all.insert_one(doc)
    raise Exception('Could do the train and test split here, too')


def carstens_train_test_split():
    db = mongoi.CarstensDb()
    id = 0
    all = db.all.find_all()
    while id < 3500:
        doc = next(all)
        id += 1
        db.train.insert_one(doc)
    while id < 4058:
        doc = next(all)
        id += 1
        db.test.insert_one(doc)


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


def remove_no_gold_label_samples():
    no_gold_label_ids = util.load_pickle('no_gold_label_ids.pkl')
    for collection in mongoi.COLLECTIONS['snli']:
        repository = mongoi.get_repository('snli', collection)
        for id in no_gold_label_ids[collection]:
            repository.delete_one(id)


NULL_VECTOR = util.load_pickle('NULL_glove_vector.pkl')


def prepend_null(sentence_matrix):
    return np.vstack([NULL_VECTOR, sentence_matrix])


def prepend_nulls(collections=mongoi.COLLECTIONS['snli']):
    for collection in collections:
        repository = mongoi.get_repository('snli', collection)
        for doc in repository.find_all():
            premise = mongoi.string_to_array(doc['premise'])
            premise = prepend_null(premise)
            repository.update_one(doc['_id'], {'premise': mongoi.array_to_string(premise)})
            hypothesis = mongoi.string_to_array(doc['hypothesis'])
            hypothesis = prepend_null(hypothesis)
            repository.update_one(doc['_id'], {'hypothesis': mongoi.array_to_string(hypothesis)})


if __name__ == '__main__':
    prepend_nulls()
