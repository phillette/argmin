from process_data import test_doc
import numpy as np
from rnn_encoders import AdditiveSentence
import networkx as nx
import matplotlib.pyplot as plt
from prediction import predict


"""
I might think about having a probability threshold for a classification - not just argmax.
E.g. the probability for a classification has to have a strength like > 0.6
It could be [0.3, 0.3, 0.4].  This is not a strong prediction.
Need to find a way to only take strong and certain predictions.
If I had a nice test set I could tune this threshold.
For now I can just play with it.
"""


THRESHOLD = 0.7


if __name__ == '__main__':
    sents, matrices = test_doc()
    num_sents = len(sents)
    adj_attack = np.zeros((num_sents, num_sents), dtype='int')
    adj_support = np.zeros((num_sents, num_sents), dtype='int')
    model = AdditiveSentence()
    for i in range(num_sents):
        for j in range(num_sents):
            premise = matrices[i][np.newaxis, ...]
            hypothesis = matrices[j][np.newaxis, ...]
            prediction = predict(model, premise, hypothesis)
            label = prediction.label(THRESHOLD)
            if label != 'neutral':
                print('Premise:')
                print(sents[i].encode('utf-8'))
                print('Hypothesis:')
                print(sents[j].encode('utf-8'))
                print('Prediction:')
                print(prediction.label(THRESHOLD))
                print('\n')
            #if prediction.encoding == 1:
            #    adj_support[i][j] = 1
            #elif prediction.encoding == 2:
            #    adj_attack[i][j] = 1
    #G = nx.from_numpy_matrix(adj_attack)
    #nx.draw(G)
    #plt.show()
