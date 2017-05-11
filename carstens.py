import mongoi
import pandas as pd
import util
import numpy as np
import stats


"""
1. import_carstens()
2. process_data.generate_friendly_ids('carstens')
3. process_data.generate_label_encodings('carstens')
4. process_data.generate_sentence_matrices('carstens')
"""


"""
Note that to split the Carstens data into train and test,
I used a variant on this code:
http://stackoverflow.com/questions/27039083/mongodb-move-documents-from-one-collection-to-another-collection
First 3500 into train, the next 558 into test, as the _id goes from 1 to 4058.
Use $gt: 3500 and $lt: 5301.
It would also be simple (simpler perhaps)
to modify the carstens_into_mongo function below.
"""


"""
Training word vectors.
I think it could be worthwhile to crawl for data on the topics in the corpus.
Generate a much larger text corpus in an attempt to get better word vectors.
So use scrapy or something?
I can generate the dictionary of words I want from the oov list I already have.
So:
1. Grab a very large collection of documents
2. Determine a dictionary for ALL words in those documents
3. Where GloVes exist, pop them into the parameter matrix
4. Where they don't, random init (for non-corpus OOV, too)
5. Train the whole thing together
6. Now have a good set of vectors (in theory)

* could also compare this approach to: random init and projection
  AND simple projection
"""


def rebuild_text_corpus():
    print("Rebuilding text from Carsten's corpus for word vector training...")
    db = mongoi.get_db('carstens')
    sentences = []
    for doc in db.all.find_all():
        sentences.append(doc['sentence1'])
        sentences.append(doc['sentence2'])
    sentences = set(sentences)
    print('Found %s unique sentences' % len(sentences))
    corpus = ' '.join(sentences)
    util.save_pickle(corpus, 'carstens_text_corpus.pkl')
    print('The corpus has %s characters' % len(corpus))
    print('Pickle saved successfully.')


def import_carstens(file_path='/home/hanshan/carstens.csv'):
    print('Importing Carstens corpus from %s' % file_path)
    X = pd.read_csv(file_path, header=None)
    label = {
        'n': 'neutral',
        's': 'entailment',
        'a': 'contradiction'
    }
    db = mongoi.CarstensDb()
    for x in X.iterrows():
        doc = {
            'sentence1': x[1][3],
            'sentence2': x[1][4],
            'gold_label': label[x[1][5]]
        }
        db.all.insert_one(doc)
    print('Carstens corpus imported successfully.')


def train_test_split(test_set_size=558):
    print('Creating train-test split for Carstens randomly...')
    db = mongoi.CarstensDb()
    ids = np.arange(stats.COLLECTION_SIZE['carstens']['all'])
    test_set_ids = np.random.choice(a=ids,
                                    size=test_set_size,
                                    replace=False)
    for doc in db.all.find_all():
        if doc['id'] in test_set_ids:
            db.test.insert_one(doc)
        else:
            db.train.insert_one(doc)
    print('Completed successfully.')
