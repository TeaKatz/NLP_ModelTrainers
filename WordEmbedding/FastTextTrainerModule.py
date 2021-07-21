import torch

from ..BaseTrainerModule import BaseTrainerModule


class FastTextTrainerModule(BaseTrainerModule):
    def __init__(self, word_embedding):
        super().__init__()
        self.word_embedding = word_embedding

    def forward(self, targets, contexts, negatives):
        """
        targets: (batch_size, ngrams_num)
        contexts: (batch_size, ngrams_num)
        negatives: (batch_size, negatives_num, ngrams_num)
        """
        # (batch_size, embedding_dim)
        targets_vec = self.word_embedding(targets)
        # (batch_size, embedding_dim)
        contexts_vec = self.word_embedding(contexts)
        # (batch_size, negatives_num, embedding_dim)
        negatives_vec = self.word_embedding(negatives)
        return targets_vec, contexts_vec, negatives_vec

    @staticmethod
    def entropy_loss_func(vec1, vec2):
        """
        vec1: (batch_size, embedding_dim)
        vec2: (batch_size, embedding_dim)
        """
        pass

    def loss_func(self, targets_vec, contexts_vec, negatives_vec):
        """
        targets_vec: (batch_size, embedding_dim)
        contexts_vec: (batch_size, embedding_dim)
        negatives_vec: (batch_size, negatives_num, embedding_dim)
        """
        negatives_num = negatives_vec.shape[1]

        positive_loss = self.entropy_loss_func(targets_vec, contexts_vec)
        negative_loss = torch.sum([self.entropy_loss_func(targets_vec, negatives_vec[:, i]) for i in range(negatives_num)], dim=0)
        return positive_loss + negative_loss

    def cal_loss(self, outputs, targets=None):
        """
        outputs: (batch_size, vocab_size)
        targets: None
        """
        targets_vec, contexts_vec, negatives_vec = outputs

        targets_vec = targets_vec.float()
        contexts_vec = contexts_vec.float()
        negatives_vec = negatives_vec.float()
        return self.loss_func(targets_vec, contexts_vec, negatives_vec)

    def cal_metrics(self, outputs, targets=None):
        """
        outputs: (batch_size, vocab_size)
        targets: None
        """
        return None
