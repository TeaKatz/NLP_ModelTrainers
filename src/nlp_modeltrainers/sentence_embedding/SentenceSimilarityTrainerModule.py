from nlp_losses import Losses
from nlp_metrics import Metrics
from ..BaseTrainerModule import BaseTrainerModule


class SentenceSimilarityTrainerModule(BaseTrainerModule):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @staticmethod
    def loss_func(outputs, targets):
        return Losses(["MAELoss"])(outputs, targets)

    def cal_loss(self, outputs, targets):
        """
        outputs: (batch_size, )
        targets: (batch_size, )
        """
        outputs = outputs.float()
        targets = targets.float()
        return self.loss_func(outputs, targets)

    def cal_metrics(self, outputs, targets):
        pass
