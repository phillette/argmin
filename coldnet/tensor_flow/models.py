"""TensorFlow base models."""
import tensorflow as tf
from coldnet import models
from coldnet.tensor_flow import util as tf_util
from coldnet.tensor_flow import blocks


class TensorFlowModel(models.Model):
    """Base model for TensorFlow models.

    Attributes:
      in_training: Boolean indicating whether in training. Set to False by
        default. The methods eval() and train() will set this externally.
        This mirrors the pytorch Module class.
    """

    def __init__(self, config):
        """Create a new TensorFlowModelBase.

        Args:
          config: coldnet.models.Config object with config settings.
        """
        super(TensorFlowModel, self).__init__(
            config=config,
            framework='tf')
        # Generate keep probability dict from config
        self.p_keep = {}
        for key in [k for k in config.keys() if k.startswith('p_keep_')]:
            self.p_keep[key.split('_')[-1]] = config[key]
        self.in_training = False

    def biases(self):
        return [v for
                v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                if v.name.endswith('biases:0')]

    def _init_backend(self):
        # Initializes all common backend parts of the computation graph
        # define on this class. It is convenient to call this at the end
        # of the constructor on a child class.
        self.logits
        self.loss
        self.optimize
        self.predictions
        self.correct_predictions
        self.accuracy
        self.confidences

    @tf_util.define_scope
    def accuracy(self):
        return tf.reduce_mean(tf.cast(self.correct_predictions, tf.float32))

    @tf_util.define_scope
    def confidences(self):
        return tf.reduce_max(self.logits, axis=1)

    @tf_util.define_scope
    def correct_predictions(self):
        return tf.equal(self.predictions, tf.argmax(self.labels, axis=0))

    def dropout_feed_dict(self):
        dropout_keys = [key for key in self.config.keys()
                        if key.startswith('p_keep_')]
        if self.in_training:
            return {getattr(self, key): self.config[key]
                    for key in dropout_keys}
        else:
            return {getattr(self, key): 1.0
                    for key in dropout_keys}

    def eval(self):
        self.in_training = False

    @tf_util.define_scope
    def labels(self):
        # default implementation safely this.
        return tf.placeholder(
            tf.int32,
            [None],
            name='labels')

    @tf_util.define_scope
    def logits(self):
        raise NotImplementedError

    @tf_util.define_scope
    def loss(self):
        return blocks.cross_entropy_with_l2(
            labels=self.labels,
            logits=self.logits,
            _lambda=self._lambda,
            parameters=self.parameters())

    @tf_util.define_scope
    def optimize(self):
        blocks.adam_with_grad_clip(
            learning_rate=self.learning_rate,
            loss=self.loss,
            parameters=self.parameters(),
            grad_clip_norm=self.grad_clip_norm)

    def parameters(self):
        return self.biases() + self.weights()

    @tf_util.define_scope
    def predictions(self):
        return tf.argmax(self.logits, axis=1)

    def train(self):
        self.in_training = True

    def weights(self):
        return [v for
                v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                if v.name.endswith('weights:0')]
