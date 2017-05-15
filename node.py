import mongoi
import xml.etree.ElementTree as ET


"""
Questions:
- What are teh attacks in Deb?
- What papers have the baselines?
- How did the baselines apportion training and testing data?
- How should I apportion training and testing data?
-----
Cabrio and Villata's 2013 paper break down a train and test split.
It is that which I will follow below.
The split is by topic - see the dicts.
"""


DATA_DIR = 'data/node/'
FILE_NAMES = [
    # DebatePedia
    'debatepediaExtended.xml',
    'extended_attacks.xml',  # what are these attack files? Labels where?
    'mediated_attacks.xml',
    'secondary_attacks.xml',
    'supported_attacks.xml',
    # 12 Angry Men
    '12AngryMen_final_dataset.xml',
    # Wikipedia
    'wiki_train.xml',
    'wiki_test.xml'
]
TEST_TOPICS = [
    'Groundzeromosque',
    'Militaryservice',
    'Noflyzone',
    'Securityprofiling',
    'Solarenergy',
    'Gasvehicles',
    'Cellphones',
    'Marijuanafree',
    'Gaymarriage',
    'Vegetarianism'
]
TRAIN_TOPICS = [
    'Violentgames',
    'Chinaonechildpolicy',
    'Cocanarcotic',
    'Childbeautycontests',
    'Arminglibianrebels',
    'Sobrietytest',
    'Osamaphoto',
    'Privatizingsocialsecurity',
    'Internetaccess'
]
LABEL = {
    'YES': 'entailment',
    'NO': 'contradiction'
}


def import_debate():
    print('Importing Node debate data into mongo...')
    db = mongoi.NodeDb()
    train, test = scrape_debate()
    for doc in train:
        db.debate_train.insert_one(doc)
    for doc in test:
        db.debate_test.insert_one(doc)
    print('Completed successfully.')


def import_wiki():
    """Import the Node Wiki data into mongo."""
    print('Importing Node wiki data into mongo...')
    db = mongoi.NodeDb()
    train_data = scrape_file(DATA_DIR + FILE_NAMES[6])
    test_data = scrape_file(DATA_DIR + FILE_NAMES[7])
    for doc in train_data:
        doc['gold_label'] = LABEL[doc['gold_label']]
        db.wiki_train.add(doc)
    for doc in test_data:
        doc['gold_label'] = LABEL[doc['gold_label']]
        db.wiki_test.add(doc)
    print('Completed successfully.')


def scrape_debate():
    data = scrape_file(DATA_DIR + FILE_NAMES[0])
    train = []
    test = []
    for doc in data:
        if doc['topic'] in TRAIN_TOPICS:
            train.append(doc)
        elif doc['topic'] in TEST_TOPICS:
            test.append(doc)
        else:
            raise Exception('Unexpected topic: %s' % doc['topic'])
    return train, test


def scrape_file(file_path):
    data = []
    tree = ET.parse(file_path)
    root = tree.getroot()
    for child in root:  # each child is a pair
        sample = {}
        sample['gold_label'] = child.attrib['entailment']
        sample['sentence1'] = child[0].text
        sample['sentence2'] = child[1].text
        if 'topic' in child.attrib.keys():
            sample['topic'] = child.attrib['topic']
        data.append(sample)
    return data
