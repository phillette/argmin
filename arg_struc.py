from process_data import test_doc
import numpy as np
from align1 import Model
import networkx as nx
import matplotlib.pyplot as plt


if __name__ == '__main__':
    sents = test_doc()
    num_sents = len(sents)
    adj_attack = np.zeros((num_sents, num_sents), dtype='int')
    adj_support = np.zeros((num_sents, num_sents), dtype='int')
    model = Model()
    for i in range(num_sents):
        for j in range(num_sents):
            premise = sents[i][np.newaxis, ...]
            hypothesis = sents[j][np.newaxis, ...]
            label = model.predict(premise, hypothesis)
            print(label)
            print(label.shape)
            if label == 1:
                adj_support[i][j] = 1
            elif label == 2:
                adj_attack[i][j] = 1
    G = nx.from_numpy_matrix(adj_attack)
    nx.draw(G)
    plt.show()
