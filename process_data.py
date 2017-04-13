from mongoi import SNLIDb, array_to_string, string_to_array, CarstensDb
import spacy
import numpy as np
import itertools
import pandas as pd
from batching import pad_sentence
from util import save_pickle, load_pickle


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
    db = SNLIDb()
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
    db = SNLIDb()
    index = 0
    for doc in db.repository(collection).find_all():
        index += 1
        premise = sentence_matrix(doc['sentence1'], nlp)
        hypothesis = sentence_matrix(doc['sentence2'], nlp)
        db.repository(collection).update_one(doc['_id'], {'premise': array_to_string(premise),
                                                          'hypothesis': array_to_string(hypothesis)})
        print(index)


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


def no_gold_labels():
    db = SNLIDb()
    query_train = db.train.find({'gold_label': '-'})
    train_ids = []
    for doc in query_train:
        train_ids.append(doc['_id'])
    save_pickle(train_ids, 'train_no_gold_label_ids.pkl')
    return train_ids


def has_missing_vector(doc, zero_vector):
    for word_vector in string_to_array(doc['premise']).tolist():
        return np.array_equal(word_vector, zero_vector)


def missing_word_vectors():
    nlp = spacy.load('en')
    zero_vector = np.zeros((300,), dtype='float')
    db = SNLIDb()
    doc_ids = []
    for doc in db.train.find_all():
        if has_missing_vector(doc, zero_vector):
            doc_ids.append(doc['_id'])
    for doc in db.dev.find_all():
        if has_missing_vector(doc, zero_vector):
            doc_ids.append(doc['_id'])
    for doc in db.test.find_all():
        if has_missing_vector(doc, zero_vector):
            doc_ids.append(doc['_id'])
    doc_ids = np.array(doc_ids)
    np.save('missing_word_vector_doc_ids.npy', doc_ids)


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
    db = CarstensDb()
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


def carstens_train_test_split():
    db = CarstensDb()
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


if __name__ == '__main__':
    no_gold_labels()
