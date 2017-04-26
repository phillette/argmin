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


"""
Process from start to finish:
1. Import data into mongo
    mongoimport --db snli --collection train "/home/hanshan/PycharmProjects/argmin/data/....jsonl"
    etc
2. remove_no_gold_label_samples()
3. generate_friendly_ids()
    - this will be an extra field this time
3. generate_sentence_matrices()
    - turn sentences into spacy docs
    - grab GloVe vectors from spacy into matrices
    - put in place OOV random vectors
    - prepend NULLs
"""


NULL_VECTOR = util.load_pickle('NULL_glove_vector.pkl')
OOV_VECTORS = util.load_pickle('oov_vectors.pkl')


def remove_no_gold_label_samples():
    print('Removing no gold label samples...')
    for collection in mongoi.COLLECTIONS['snli']:
        print('Deleting no gold labels from collection: "%s"' % collection)
        deleted_count = 0
        repository = mongoi.get_repository('snli', collection)
        for doc in repository.find_all():
            if doc['gold_label'] == '-':
                repository.delete_one(doc['_id'])
                deleted_count += 1
        print('Deleted %s documents' % deleted_count)
        print('Remaining count in collection: %s' % repository.count())
    print('Finished successfully.')


def generate_friendly_ids():
    print('Generating friendly ids...')
    id = 1
    for collection in mongoi.COLLECTIONS['snli']:
        print('Working on collection: %s' % collection)
        repository = mongoi.get_repository('snli', collection)
        for doc in repository.find_all():
            repository.update_one(doc['_id'], {'id': id})
            id += 1
    print('Completed successfully.')


def generate_sentence_matrices():
    print('Generating sentence matrices...')
    nlp = spacy.load('en')
    for collection in mongoi.COLLECTIONS['snli']:
        print('Working on collection: %s' % collection)
        repository = mongoi.get_repository('snli', collection)
        for doc in repository.find_all():
            premise = sentence_matrix(doc['sentence1'], nlp)
            hypothesis = sentence_matrix(doc['sentence2'], nlp)
            repository.update_one(doc['_id'], {'premise': mongoi.array_to_string(premise),
                                               'hypothesis': mongoi.array_to_string(hypothesis)})
    print('Completed successfully.')


def sentence_matrix(sentence, nlp):
    doc = nlp(sentence)
    matrix = np.vstack(list(get_vector(t).reshape((1, 300)) for t in doc))
    return prepend_null(matrix)


def get_vector(token):
    if token.text in OOV_VECTORS.keys():
        return OOV_VECTORS[token.text]
    else:
        return token.vector


def prepend_null(sentence_matrix):
    return np.vstack([NULL_VECTOR, sentence_matrix])


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
    remove_no_gold_label_samples()
    generate_friendly_ids()
    generate_sentence_matrices()
