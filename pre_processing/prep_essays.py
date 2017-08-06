"""Argumentation mining of persuasive essay corpus."""
import numpy as np


"""
Experiments:
A) Carstens-style sentence comparison
B) Essay-to-tree end-to-end AM

Plan:
1) Prepare the data
    a) For all .ann files, extract the nodes and their relationships, and build
       the data set of pairs from this; paying attention to train-test splits,
       and cross-validation splits.
    b) Will need constituency parses of all sentences; represent the document in
       a way that interfaces with the model (a forest of trees?)
2) Build the models
3) Assess the results
"""


NUM_ESSAYS = 402
# These are the NLI data labels
REVERSE_LABEL_MAP = {
    0: "entailment",
    1: "neutral",
    2: "contradiction"}
# These are in line with NLI labels
STANCE_CODE = {
    'For': 0,
    'none': 1,
    'Against': 2}
RELATION_CODE = {
    'supports': 0,
    'attacks': 2}


class Essay:
    def __init__(self, essay_no):
        self.essay_no = essay_no
        file_path = get_file_path(essay_no)
        node_lines, stance_lines, relation_lines = lines_by_type(file_path)
        nodes, ids_to_ixs, ixs_to_ids = text_nodes(essay_no, node_lines)
        self.nodes = nodes
        self.major_claim_ixs = get_major_claim_ixs(nodes)
        self.adj_mat = get_adj_mat(
            nodes, ids_to_ixs, stance_lines, relation_lines)


class TextNode:
    def __init__(self, essay_no, id, ix, type, text):
        self.essay_no = essay_no
        self.id = id
        self.ix = ix
        self.type = type
        self.text = text

    def __repr__(self):
        return 'essay%s\nT%s\n(%s)\nType "%s"\nText:\n%s' % (self.essay_no,
                                                             self.id,
                                                             self.ix,
                                                             self.type,
                                                             self.text)


class Relations:
    def __init__(self, essay_number, node_count, stance_lines, relation_lines):
        self.essay_number = essay_number
        self.adj_mat = np.ones((node_count, node_count), dtype='int32')
        for stance_line in stance_lines:
            split_line = stance_line.rstrip().split('\t')[1].split(' ')
            child_id = int(split_line[1].replace('T', '')) - 1
            relation = STANCE_CODE[split_line[2]]
            self.adj_mat[0][child_id] = relation
        for relation_line in relation_lines:
            split_line = relation_line.rstrip().split('\t')[1].split(' ')
            child_id = int(split_line[1].split('T')[1]) - 1
            parent_id = int(split_line[2].split('T')[1]) - 1
            relation = RELATION_CODE[split_line[0]]
            self.adj_mat[parent_id][child_id] = relation


def build_corpus_a():
    corpus = []
    for essay in get_essays():
        corpus += essay_to_labeled_pairs(essay)
    return corpus


def essay_to_labeled_pairs(essay):
    data = []
    for s1 in essay.nodes:
        for s2 in essay.nodes:
            # Come back and double check this logic: which the child, parent?
            sample = {
                'sentence1': s1.text,
                'sentence2': s2.text,
                'gold_label': REVERSE_LABEL_MAP[essay.adj_mat[s2.ix][s1.ix]],
                'label': essay.adj_mat[s2.ix][s1.ix]}
            data.append(sample)
    return data


def get_adj_mat(nodes, ids_to_ixs, stance_lines, relation_lines):
    node_count = len(nodes)
    major_claim_ixs = get_major_claim_ixs(nodes)
    adj_mat = np.ones((node_count, node_count), dtype='int32')
    for stance_line in stance_lines:
        split_line = stance_line.rstrip().split('\t')[1].split(' ')
        child_id = split_line[1]
        child_ix = ids_to_ixs[child_id]
        relation = STANCE_CODE[split_line[2]]
        for parent_ix in major_claim_ixs:
            adj_mat[parent_ix][child_ix] = relation
    for relation_line in relation_lines:
        split_line = relation_line.rstrip().split('\t')[1].split(' ')
        child_id = split_line[1].split(':')[1]
        child_ix = ids_to_ixs[child_id]
        parent_id = split_line[2].split(':')[1]
        parent_ix = ids_to_ixs[parent_id]
        relation = RELATION_CODE[split_line[0]]
        adj_mat[parent_ix][child_ix] = relation
    return adj_mat


def get_essay_no(file_path):
    file_name = file_path.split('/')[-1]
    essay_no = file_name.replace('.ann', '').replace('essay', '')
    return essay_no


def get_essay_nos():
    return [i + 1 for i in range(NUM_ESSAYS)]


def get_essays():
    return [Essay(no) for no in get_essay_nos()]


def get_file_path(essay_no):
    if essay_no >= 100:
        no_str = '%s' % essay_no
    elif essay_no >= 10:
        no_str = '0%s' % essay_no
    else:
        no_str = '00%s' % essay_no
    return 'data/pec/essay%s.ann' % no_str


def get_major_claim_ixs(nodes):
    return [n.ix for n in nodes if n.type == 'MajorClaim']


def lines_by_type(file_path):
    with open(file_path) as file:
        lines = file.readlines()
        node_lines = [l for l in lines if l.startswith('T')]
        stance_lines = [l for l in lines if l.startswith('A')]
        relation_lines = [l for l in lines if l.startswith('R')]
    return node_lines, stance_lines, relation_lines


def text_nodes(essay_no, node_lines):
    split_lines = [l.split('\t') for l in node_lines]
    ids = [l[0] for l in split_lines]
    ixs = list(range(len(node_lines)))
    types = [l[1].split(' ')[0] for l in split_lines]
    texts = [l[2] for l in split_lines]
    nodes = [TextNode(essay_no, ids[i], ixs[i], types[i], texts[i])
             for i in range(len(node_lines))]
    ids_to_ixs = dict(zip(ids, ixs))
    ixs_to_ids = dict(zip(ixs, ids))
    return nodes, ids_to_ixs, ixs_to_ids


def view_file(essay_no):
    with open(get_file_path(essay_no)) as f:
        lines = f.readlines()
        for line in lines:
            print(line)


if __name__ == '__main__':
    corpus = build_corpus_a()
    print(len(corpus))
    print(len([s for s in corpus if s['label'] == 0]))
    print(len([s for s in corpus if s['label'] == 1]))
    print(len([s for s in corpus if s['label'] == 2]))


# Corpus A stats:
# 98471 sentence pairs
# 5958 supports
# 91798 neutrals
# 715 attacks
