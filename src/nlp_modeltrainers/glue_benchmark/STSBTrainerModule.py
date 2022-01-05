import torch
from nlp_losses import Losses
from nlp_metrics.glue_benchmark import STSBMetric
from ..BaseTrainerModule import BaseTrainerModule


class STSBTrainerModule(BaseTrainerModule):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        outputs = torch.nn.CosineSimilarity(dim=1)(outputs[:, 0], outputs[:, 1])
        return outputs

    @staticmethod
    def loss_func(outputs, targets):
        return Losses(["MSELoss"])(outputs, targets)

    @staticmethod
    def metrics_func(outputs, targets):
        return STSBMetric()(outputs, targets)

    def cal_loss(self, outputs, targets):
        """
        outputs: (batch_size, )
        targets: (batch_size, )
        """
        outputs = outputs.float()
        targets = targets.float()
        return self.loss_func(outputs, targets)

    def cal_metrics(self, outputs, targets):
        """
        outputs: (batch_size, n_classes)
        targets: (batch_size, )
        """
        outputs = outputs.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()

        outputs = outputs.astype(float)
        targets = targets.astype(float)
        return self.metrics_func(outputs, targets)
