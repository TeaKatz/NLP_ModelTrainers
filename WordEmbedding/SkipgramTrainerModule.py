import torch

from torch.nn import BCEWithLogitsLoss, Linear

from ...Metrics import Metrics
from ..BaseTrainerModule import BaseTrainerModule


class SkipgramTrainerModule(BaseTrainerModule):
    def __init__(self, word_embedding, embedding_dim, vocab_size):
        super().__init__()
        self.word_embedding = word_embedding
        self.context_classifier = Linear(embedding_dim, vocab_size)

    def forward(self, *args, **kwargs):
        # (batch_size, 1, embedding_dim)
        outputs = self.word_embedding(*args, **kwargs)
        # (batch_size, embedding_dim)
        outputs = torch.squeeze(outputs, dim=1)
        # (batch_size, vocab_size)
        outputs = self.context_classifier(outputs)
        return outputs

    @staticmethod
    def loss_func(outputs, targets):
        return BCEWithLogitsLoss()(outputs, targets)

    @staticmethod
    def metrics_func(outputs, targets):
        return Metrics(["MultiLabel-F1", "MultiLabel-Precision", "MultiLabel-Recall"], names=["F1", "Precision", "Recall"], average="micro", zero_division=0)(outputs, targets)

    def cal_loss(self, outputs, targets):
        """
        outputs: (batch_size, vocab_size)
        targets: (batch_size, vocab_size)
        """
        targets = targets.long()
        outputs = outputs.float()
        return self.loss_func(outputs, targets)

    def cal_metrics(self, outputs, targets):
        """
        outputs: (batch_size, vocab_size)
        targets: (batch_size, vocab_size)
        """
        targets = targets.cpu().detach().numpy()
        outputs = outputs.cpu().detach().numpy()

        targets = targets.astype(int)
        outputs = (outputs > 0).astype(int)
        return self.metrics_func(outputs, targets)


class DualSkipgramTrainerModule(BaseTrainerModule):
    def __init__(self, word_embedding1, word_embedding2, embedding_dim, vocab_size):
        super().__init__()
        self.word_embedding1 = word_embedding1
        self.word_embedding2 = word_embedding2
        self.context_classifier = Linear(embedding_dim, vocab_size)

    def forward(self, inputs1, inputs2):
        # (batch_size, 1, embedding_dim)
        vecs1 = self.word_embedding1(inputs1)
        vecs2 = self.word_embedding2(inputs2)
        # (batch_size, embedding_dim)
        vecs1 = torch.squeeze(vecs1, dim=1)
        vecs2 = torch.squeeze(vecs2, dim=1)
        # (batch_size, vocab_size)
        outputs1 = self.context_classifier(vecs1)
        outputs2 = self.context_classifier(vecs2)
        return outputs1, outputs2, vecs1, vecs2

    @staticmethod
    def loss_func(outputs, targets):
        return BCEWithLogitsLoss()(outputs, targets)

    @staticmethod
    def regularize_func(vecs1, vecs2):
        # (batch_size, 1, embedding_dim)
        vecs1 = torch.unsqueeze(vecs1, dim=1)
        vecs2 = torch.unsqueeze(vecs2, dim=1)
        # (batch_size, 1, 1)
        dot_product = torch.matmul(vecs1, vecs2.transpose(1, 2))
        logistic_loss = torch.log(1 + torch.exp(-dot_product))
        return torch.sum(logistic_loss)

    def cal_loss(self, outputs, targets):
        """
        outputs: (outputs1, outputs2, vecs1, vecs2)
        outputs1, outputs2: (batch_size, vocab_size)
        vecs1, vecs2: (batch_size, embedding_dim)
        targets: (batch_size, vocab_size)
        """
        outputs1, outputs2, vecs1, vecs2 = outputs

        targets = targets.long()
        outputs1 = outputs1.float()
        outputs2 = outputs2.float()
        vecs1 = vecs1.float()
        vecs2 = vecs2.float()
        return self.loss_func(outputs1, targets) + self.loss_func(outputs2, targets) + self.regularize_func(vecs1, vecs2)

    def cal_metrics(self, outputs, targets):
        pass


class VocabFreeSkipgramTrainerModule(BaseTrainerModule):
    def __init__(self, word_embedding):
        super().__init__()
        self.word_embedding = word_embedding

    def forward(self, targets, contexts, negatives):
        """
        targets: (*, embedding_dim)
        contexts: (*, embedding_dim)
        negatives: (*, negatives_num, embedding_dim)
        """
        # (batch_size, 1, embedding_dim)
        target_outputs = self.word_embedding(targets)
        context_outputs = self.word_embedding(contexts)
        # (batch_size, negatives_num, embedding_dim)
        negative_outputs = self.word_embedding(negatives)
        return target_outputs, context_outputs, negative_outputs
        

    @staticmethod
    def loss_func(outputs, targets):
        return BCEWithLogitsLoss()(outputs, targets)

    @staticmethod
    def metrics_func(outputs, targets):
        return Metrics(["MultiLabel-F1", "MultiLabel-Precision", "MultiLabel-Recall"], names=["F1", "Precision", "Recall"], average="micro", zero_division=0)(outputs, targets)

    def cal_loss(self, outputs, targets):
        """
        outputs: (batch_size, vocab_size)
        targets: (batch_size, vocab_size)
        """
        targets = targets.long()
        outputs = outputs.float()
        return self.loss_func(outputs, targets)

    def cal_metrics(self, outputs, targets):
        """
        outputs: (batch_size, vocab_size)
        targets: (batch_size, vocab_size)
        """
        targets = targets.cpu().detach().numpy()
        outputs = outputs.cpu().detach().numpy()

        targets = targets.astype(int)
        outputs = (outputs > 0).astype(int)
        return self.metrics_func(outputs, targets)
