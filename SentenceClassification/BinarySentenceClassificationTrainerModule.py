import torch
from torch.nn import BCEWithLogitsLoss

from ...Metrics import Metrics
from ..BaseTrainerModule import BaseTrainerModule


class BinarySentenceClassificationTrainerModule(BaseTrainerModule):
    @staticmethod
    def loss_func(outputs, targets):
        return BCEWithLogitsLoss()(outputs, targets)

    @staticmethod
    def metrics_func(outputs, targets):
        return Metrics(["F1", "Precision", "Recall"], names=["F1", "Precision", "Recall"], pos_label=1, average="binary", zero_division=0)(outputs, targets)

    def cal_loss(self, outputs, targets):
        """
        targets: (batch_size, )
        outputs: (batch_size, )
        """
        targets = targets.long()
        outputs = outputs.float()
        return self.loss_func(outputs, targets)

    def cal_metrics(self, outputs, targets):
        """
        targets: (batch_size, )
        outputs: (batch_size, )
        """
        targets = targets.cpu().detach().numpy()
        outputs = outputs.cpu().detach().numpy()

        targets = targets.astype(int)
        outputs = (outputs > 0).astype(int)
        return self.metrics_func(outputs, targets)
