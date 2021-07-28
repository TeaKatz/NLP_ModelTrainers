import numpy as np

from ...NLP_Losses import Losses
from ...NLP_Metrics import Metrics
from ..BaseTrainerModule import BaseTrainerModule


class MulticlassSentenceClassificationTrainerModule(BaseTrainerModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @staticmethod
    def loss_func(outputs, targets):
        return Losses(["CategoricalCrossEntropyLoss"])(outputs, targets)

    @staticmethod
    def metrics_func(outputs, targets):
        return Metrics(["F1", "Precision", "Recall"], average="micro", zero_division=0)(outputs, targets)

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
