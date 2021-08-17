from nlp_losses import Losses
from nlp_metrics import Metrics
from ..BaseTrainerModule import BaseTrainerModule


class Vector2TokensTrainerModule(BaseTrainerModule):
    def __init__(self, word_embedding, classification_head):
        super().__init__()
        self.word_embedding = word_embedding
        self.classification_head = classification_head

    def forward(self, word_embedding_in, classification_head_in):
        if isinstance(word_embedding_in, dict):
            word_embedding_out = self.word_embedding(**word_embedding_in)
        else:
            word_embedding_out = self.word_embedding(word_embedding_in)

        if isinstance(classification_head_in, dict):
            classification_head_out = self.classification_head(word_embedding_out, **classification_head_in)
        else:
            classification_head_out = self.classification_head(word_embedding_out, classification_head_in)
        return classification_head_out

    @staticmethod
    def loss_func(outputs, targets):
        return Losses("CategoricalCrossEntropyLoss", ignore_index=0)(outputs, targets)

    def cal_loss(self, outputs, targets):
        """
        outputs: (batch_size, *, class_size)
        targets: (batch_size, *)
        """
        class_size = outputs.shape[-1]

        outputs = outputs.float()
        targets = targets.long()

        outputs = outputs.view(-1, class_size)
        targets = targets.view(-1)
        return self.loss_func(outputs, targets)

    def cal_metrics(self, outputs, targets):
        pass