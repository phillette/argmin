import mongoi
import spacy
import process_data
import pandas as pd

"""
Note that to split the Carstens data into train and test,
I used a variant on this code:
http://stackoverflow.com/questions/27039083/mongodb-move-documents-from-one-collection-to-another-collection
First 3500 into train, the next 558 into test, as the _id goes from 1 to 4058.
Use $gt: 3500 and $lt: 5301.
It would also be simple (simpler perhaps)
to modify the carstens_into_mongo function below.
"""


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
        doc['premise'] = mongoi.array_to_string(process_data.sentence_matrix(doc['sentence1'], nlp))
        doc['hypothesis'] = mongoi.array_to_string(process_data.sentence_matrix(doc['sentence2'], nlp))
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
