import torch
from nlp_losses import Losses
from nlp_metrics.glue_benchmark import STSBMetric
from ..BaseTrainerModule import BaseTrainerModule

N_CLASSES = 1


class STSBTrainerModule(BaseTrainerModule):
    def __init__(self, model, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.classifier = torch.nn.Linear(hidden_size, N_CLASSES)

    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        outputs = self.classifier(outputs)
        return outputs

    @staticmethod
    def loss_func(outputs, targets):
        return Losses(["MSELoss"])(outputs, targets)

    @staticmethod
    def metrics_func(outputs, targets):
        return STSBMetric()(outputs, targets)

    def cal_loss(self, outputs, targets):
        """
        outputs: (batch_size, n_classes)
        targets: (batch_size, )
        """
        outputs = outputs.float()
        targets = targets.float()
        # (batch_size, )
        outputs = outputs.squeeze(1)
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
        # (batch_size, )
        outputs = outputs.squeeze(1)
        return self.metrics_func(outputs, targets)
