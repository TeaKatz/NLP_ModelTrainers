import torch
from torch.nn import BCEWithLogitsLoss

from ...Metrics import Metrics
from ..BaseTrainerModule import BaseTrainerModule


class MultilabelSentenceClassificationTrainerModule(BaseTrainerModule):
    @staticmethod
    def loss_func(outputs, targets):
        return BCEWithLogitsLoss()(outputs, targets)

    @staticmethod
    def metrics_func(outputs, targets):
        return Metrics(["MultiLabel-F1", "MultiLabel-Precision", "MultiLabel-Recall"], names=["F1", "Precision", "Recall"], average="micro", zero_division=0)(outputs, targets)

    def cal_loss(self, outputs, targets):
        """
        targets: (batch_size, class_size)
        outputs: (batch_size, class_size)
        """
        targets = targets.long()
        outputs = outputs.float()
        return self.loss_func(outputs, targets)

    def cal_metrics(self, outputs, targets):
        """
        targets: (batch_size, class_size)
        outputs: (batch_size, class_size)
        """
        targets = targets.cpu().detach().numpy()
        outputs = outputs.cpu().detach().numpy()

        targets = targets.astype(int)
        outputs = (outputs > 0).astype(int)
        return self.metrics_func(outputs, targets)
