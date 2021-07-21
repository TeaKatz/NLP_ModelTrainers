import torch
import numpy as np
from torch.nn import CrossEntropyLoss

from ...Metrics import Metrics
from ..BaseTrainerModule import BaseTrainerModule


class MulticlassSentenceClassificationTrainerModule(BaseTrainerModule):
    @staticmethod
    def loss_func(outputs, targets):
        return CrossEntropyLoss()(outputs, targets)

    @staticmethod
    def metrics_func(outputs, targets):
        return Metrics(["F1", "Precision", "Recall"], names=["F1", "Precision", "Recall"], average="micro", zero_division=0)(outputs, targets)

    def cal_loss(self, outputs, targets):
        """
        targets: (batch_size, )
        outputs: (batch_size, class_size)
        """
        targets = targets.long()
        outputs = outputs.float()
        return self.loss_func(outputs, targets)

    def cal_metrics(self, outputs, targets):
        """
        targets: (batch_size, )
        outputs: (batch_size, class_size)
        """
        targets = targets.cpu().detach().numpy()
        outputs = outputs.cpu().detach().numpy()

        targets = targets.astype(int)
        outputs = np.argmax(outputs, axis=-1).astype(int)
        return self.metrics_func(outputs, targets)
