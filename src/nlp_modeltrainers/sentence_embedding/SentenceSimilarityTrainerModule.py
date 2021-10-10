from nlp_losses import Losses
from nlp_metrics import Metrics
from ..BaseTrainerModule import BaseTrainerModule


class SentenceSimilarityTrainerModule(BaseTrainerModule):
    def __init__(self, model, learning_rate=1e-3):
        super().__init__(learning_rate)
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @staticmethod
    def loss_func(output1, output2, targets):
        return Losses(["CosineSimilarityLoss"])(output1, output2, targets)

    def cal_loss(self, outputs, targets):
        """
        outputs: 
            output1: (batch_size, vector_size)
            output2: (batch_size, vector_size)
        targets: (batch_size, )
        """
        output1, output2 = outputs

        output1 = output1.float()
        output2 = output2.float()
        targets = targets.float()
        return self.loss_func(output1, output2, targets)

    def cal_metrics(self, outputs, targets):
        pass
