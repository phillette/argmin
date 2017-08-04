"""Argumentation mining of persuasive essay corpus."""
import os
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


STANCE_CODE = {
    'For': 0,
    'none': 1,
    'Against': 2}
RELATION_CODE = {
    'supports': 0}


class TextNode:
    def __init__(self, essay_number, line):
        self.essay_number = essay_number
        self.line = line
        split_line = line.split('\t')
        self.id = int(split_line[0].replace('T', '')) - 1
        self.type = split_line[1].split(' ')[0]
        self.text = split_line[2]

    def __repr__(self):
        return 'essay%s\nT%s\nType "%s"\nText:\n%s' % (self.essay_number,
                                                       self.node_id,
                                                       self.node_type,
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
    nodes = []
    relations = []
    file_names = [f for f in os.listdir('data/pec') if f.endswith('ann')]
    for file_name in file_names:
        try:
            file_nodes, file_relations = nodes_and_relations('data/pec/%s'
                                                             % file_name)
        except Exception as e:
            print(file_name)
            raise e
    #    nodes += file_nodes
    #    relations.append(file_relations)
    #for i in range(len(nodes)):
    #    corpus.append({})


def nodes_and_relations(file_path):
    file_name = file_path.split('/')[-1]
    essay_number = file_name.replace('.ann', '').replace('essay', '')
    with open(file_path) as file:
        lines = file.readlines()
        text_nodes = [TextNode(essay_number, l) for l in lines
                      if l.startswith('T')]
        stance_lines = [l for l in lines if l.startswith('A')]
        relation_lines = [l for l in lines if l.startswith('R')]
        relations = Relations(
            essay_number, len(text_nodes), stance_lines, relation_lines)
        return text_nodes, relations


if __name__ == '__main__':
    build_corpus_a()
    #file_path = 'data/pec/essay022.ann'
    #nodes, relations = nodes_and_relations(file_path)
    #for node in nodes:
    #    print(node)
    #print(relations.adj_mat)
