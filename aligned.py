import tensorflow as tf
from models import Model
from tf_decorators import define_scope


class Alignment(Model):
    """
    Alignment model without RNN.
    """
    def __init__(self, word_embed_length=300, learning_rate=1e-2,
                 ff_hidden_size=200, p_keep_ff=0.5, grad_norm=3.0):
        self.ff_hidden_size = ff_hidden_size
        self.p_keep_ff = p_keep_ff
        self.grad_norm = grad_norm
        Model.__init__(self, word_embed_length, learning_rate)
        self.name = 'alignment'

    @define_scope
    def _alignments(self):
        passata
