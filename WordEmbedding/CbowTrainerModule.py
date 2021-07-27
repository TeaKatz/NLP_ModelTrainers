import torch
import numpy as np

from torch.nn import Linear

from ...Losses import Losses
from ...Metrics import Metrics
from ..BaseTrainerModule import BaseTrainerModule


class CbowTrainerModule(BaseTrainerModule):
    def __init__(self, word_embedding, embedding_dim, vocab_size):
        super().__init__()
        self.word_embedding = word_embedding
        self.context_classifier = Linear(embedding_dim, vocab_size)

    def forward(self, *args, **kwargs):
        # (batch_size, words_num, embedding_dim)
        outputs = self.word_embedding(*args, **kwargs)
        # (batch_size, embedding_dim)
        outputs = torch.sum(outputs, dim=1)
        # (batch_size, vocab_size)
        outputs = self.context_classifier(outputs)
        return outputs

    @staticmethod
    def loss_func(outputs, targets):
        return Losses(["CategoricalCrossEntropyLoss"])(outputs, targets)

    @staticmethod
    def metrics_func(outputs, targets):
        return Metrics(["F1", "Precision", "Recall"], average="micro", zero_division=0)(outputs, targets)

    def cal_loss(self, outputs, targets):
        """
        outputs: (batch_size, vocab_size)
        targets: (batch_size, )
        """
        targets = targets.long()
        outputs = outputs.float()
        return self.loss_func(outputs, targets)

    def cal_metrics(self, outputs, targets):
        """
        outputs: (batch_size, vocab_size)
        targets: (batch_size, )
        """
        targets = targets.cpu().detach().numpy()
        outputs = outputs.cpu().detach().numpy()

        targets = targets.astype(int)
        outputs = np.argmax(outputs, axis=-1).astype(int)
        return self.metrics_func(outputs, targets)
