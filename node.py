import mongoi
import xml.etree.ElementTree as ET
import process_data


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
    'training_set_ESWC.xml',
    'test_set_ESWC.xml'
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


def scrape_debate():
    data = scrape_file(DATA_DIR + FILE_NAMES[0])
    return data


def scrape_file(file_path):
    data = []
    tree = ET.parse(file_path)
    root = tree.getroot()
    for child in root:  # each child is a pair
        sample = {}
        sample['label'] = child.attrib['entailment'].encode('utf-8')
        sample['premise'] = child[0].text.encode('utf-8')
        sample['hypothesis'] = child[1].text.encode('utf-8')
        sample['topic'] = child.attrib['topic'].encode('utf-8')
        data.append(sample)
    return data


if __name__ == '__main__':
    data = scrape_file(DATA_DIR + FILE_NAMES[0])
    print(set(d['topic'] for d in data))
