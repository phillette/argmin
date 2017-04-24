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
    db = SNLIDb()
    train_ids = _get_missing_gold_label_ids(db, 'train')
    dev_ids = _get_missing_gold_label_ids(db, 'dev')
    test_ids = _get_missing_gold_label_ids(db, 'test')
    save_pickle(train_ids, 'train_no_gold_label_ids.pkl')
    save_pickle(dev_ids, 'dev_no_gold_label_ids.pkl')
    save_pickle(test_ids, 'test_no_gold_label_ids.pkl')
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
    words = string_to_array(doc[array_attr]).tolist()
    zeros = []
    for i in range(len(words)):
        if np.array_equal(words[i], zero_vector):
            zeros.append(spacy_doc[i].text)
    return len(zeros) > 0, zeros


def _get_missing_word_vectors():
    nlp = spacy.load('en')
    zero_vector = np.zeros((300,), dtype='float')
    db = SNLIDb()
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
    save_pickle(missing_vectors, 'missing_vectors.pkl')
    return missing_vectors


def missing_word_vectors():
    missing_vectors = load_pickle('missing_vectors.pkl')
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


def process_oov():
    mv = load_pickle('missing_vectors.pkl')
    oov = {}  # word: id
    id = 0
    all_words = mv['train']['words'].values() \
                + mv['dev']['words'].values() \
                + mv['test']['words'].values()
    for word in all_words:
        if word not in oov.keys():
            oov[word] = id
            id += 1
    save_pickle(oov, 'oov_ids.pkl')
    oov_count = len(list(oov.keys()))
    vectors = {}  # id, vector
    for _, id in oov.items():
        vector = np.random.rand(1, 300)
        vectors[id] = vector
    save_pickle(vectors, 'oov_vectors.pkl')
    # fuck, the real problem is to know in which position to insert the random vector.


if __name__ == '__main__':
    _get_missing_word_vectors()
