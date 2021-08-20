import torch

import numpy as np

from torch.nn import Linear

from nlp_losses import Losses
from nlp_metrics import Metrics
from ..BaseTrainerModule import BaseTrainerModule


class SkipgramTrainerModule(BaseTrainerModule):
    def __init__(self, word_embedding, embedding_dim, vocab_size, learning_rate=1e-3):
        super().__init__(learning_rate)
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
        return Losses(["BinaryCrossEntropyLoss"])(outputs, targets)

    @staticmethod
    def metrics_func(outputs, targets):
        return Metrics(["MultiLabel_F1", "MultiLabel_Precision", "MultiLabel_Recall"], names=["F1", "Precision", "Recall"], average="micro", zero_division=0)(outputs, targets)

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
    def __init__(self, word_embedding1, word_embedding2, embedding_dim, vocab_size, learning_rate=1e-3):
        super().__init__(learning_rate)
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
        return Losses(["BinaryCrossEntropyLoss"])(outputs, targets)

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
    def __init__(self, word_embedding, learning_rate=1e-3):
        super().__init__(learning_rate)
        self.word_embedding = word_embedding

    def forward(self, targets, contexts, negatives):
        # (batch_size, 1, embedding_dim)
        if isinstance(targets, dict):
            target_outputs = self.word_embedding(**targets)
        else:
            target_outputs = self.word_embedding(targets)

        # (batch_size, 1, embedding_dim)
        if isinstance(contexts, dict):
            context_outputs = self.word_embedding(**contexts)
        else:
            context_outputs = self.word_embedding(contexts)

        # (batch_size, negatives_num, embedding_dim)
        if isinstance(negatives, dict):
            negative_outputs = self.word_embedding(**negatives)
        else:
            negative_outputs = self.word_embedding(negatives)
        return target_outputs, context_outputs, negative_outputs

    @staticmethod
    def loss_func(anchor, positive, negative):
        return Losses(["TripletLoss"], reduction="sum")([anchor, positive, negative])

    @staticmethod
    def metrics_func(anchor, positive, negative):
        metrics_dict = {}
        # Positive cosine similarity
        metrics_dict["Positive_Cosine_Similarity"] = Metrics(["Cosine_Similarity"])(anchor[:, 0], positive[:, 0])["Cosine_Similarity"]

        # Negative cosine similarity
        neg_cosine_sim = []
        for i in range(negative.shape[1]):
            neg_cosine_sim.append(Metrics(["Cosine_Similarity"])(anchor[:, 0], negative[:, i])["Cosine_Similarity"])
        neg_cosine_sim = np.mean(neg_cosine_sim)
        metrics_dict["Negative_Cosine_Similarity"] = neg_cosine_sim
        return metrics_dict

    def cal_loss(self, outputs, targets=None):
        """
        outputs: (target_outputs, context_outputs, negative_outputs)
        target_outputs: (batch_size, 1, embedding_dim)
        context_outputs: (batch_size, 1, embedding_dim)
        negative_outputs: (batch_size, negatives_num, embedding_dim)
        """
        target_outputs, context_outputs, negative_outputs = outputs

        target_outputs = target_outputs.float()
        context_outputs = context_outputs.float()
        negative_outputs = negative_outputs.float()
        return self.loss_func(target_outputs, context_outputs, negative_outputs)

    def cal_metrics(self, outputs, targets=None):
        """
        outputs: (target_outputs, context_outputs, negative_outputs)
        target_outputs: (batch_size, 1, embedding_dim)
        context_outputs: (batch_size, 1, embedding_dim)
        negative_outputs: (batch_size, negatives_num, embedding_dim)
        """
        target_outputs, context_outputs, negative_outputs = outputs

        target_outputs = target_outputs.cpu().detach().numpy()
        context_outputs = context_outputs.cpu().detach().numpy()
        negative_outputs = negative_outputs.cpu().detach().numpy()

        target_outputs = target_outputs.astype(float)
        context_outputs = context_outputs.astype(float)
        negative_outputs = negative_outputs.astype(float)
        return self.metrics_func(target_outputs, context_outputs, negative_outputs)
