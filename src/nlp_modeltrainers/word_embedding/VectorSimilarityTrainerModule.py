from torch.nn import CosineSimilarity

from nlp_losses import Losses
from ..BaseTrainerModule import BaseTrainerModule


class VectorCosineSimilarityTrainerModule(BaseTrainerModule):
    def __init__(self, word_embedding):
        super().__init__()
        self.word_embedding = word_embedding

    def forward(self, *args, **kwargs):
        # (batch_size, 2, embedding_dim)
        vecs = self.word_embedding(*args, **kwargs)
        # (batch_size, embedding_dim)
        vec1 = vecs[:, 0]
        vec2 = vecs[:, 1]
        # (batch_size, )
        cosine = CosineSimilarity(dim=1)(vec1, vec2)
        return cosine

    @staticmethod
    def loss_func(outputs, targets):
        """
        outputs: (batch_size, )
        targets: (batch_size, )
        """
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
