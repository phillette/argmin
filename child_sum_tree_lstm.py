import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim


torch.manual_seed(1)


class ChildSumTreeLSTM(nn.Module):
    """"""

    def __init__(self, vocab_size, word_to_ix, embedding, embedding_dim,
                 hidden_dim):
        super(ChildSumTreeLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.word_to_ix = word_to_ix
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # init here for now
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        pass
