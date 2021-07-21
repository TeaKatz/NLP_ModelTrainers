import torch

from torch.nn import Module, Embedding


class PositionalCharacterLevelWordSparse(Module):
    def __init__(self, num_embeddings, padding_idx=0, max_positional=10):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.padding_idx = padding_idx
        self.max_positional = max_positional

    def forward(self, token_ids, position_ids):
        """
        token_ids: (batch_size, words_num, word_length)
        position_ids: (batch_size, words_num, word_length)

        Return
        word_vecs: (batch_size, words_num, num_embeddings)
        """
        batch_size, words_num, word_length = token_ids.shape

        word_vecs = torch.zeros(batch_size, words_num, self.num_embeddings + self.max_positional)
        for i in range(batch_size):
            for j in range(words_num):
                for k in range(word_length):
                    if token_ids[i, j, k] != self.padding_idx:
                        word_vecs[i, j, token_ids[i, j, k]] += 1
                        word_vecs[i, j, self.num_embeddings + position_ids[i, j, k]] += 1
        return word_vecs


class PositionalCharacterLevelWordEmbedding(Module):
    mode_options = ["sum", "mean", "max"]

    def __init__(self, num_embeddings, embedding_dim, padding_idx=0, max_positional=10, mode="sum"):
        assert mode in self.mode_options

        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_positional = max_positional
        self.mode = mode

        self.l_word_embedding = Embedding(num_embeddings=num_embeddings, 
                                          embedding_dim=embedding_dim, 
                                          padding_idx=padding_idx)

        self.l_pos_embedding = Embedding(num_embeddings=max_positional,
                                         embedding_dim=embedding_dim,
                                         padding_idx=padding_idx)

    def forward(self, token_ids, position_ids):
        """
        token_ids: (batch_size, words_num, word_length)
        position_ids: (batch_size, words_num, word_length)

        Return
        word_vecs: (batch_size, words_num, embedding_dim)
        """
        # (batch_size, words_num, word_length, embedding_dim)
        word_vecs = self.l_word_embedding(token_ids) + self.l_pos_embedding(position_ids)

        # (batch_size, words_num, embedding_dim)
        if self.mode == "sum":
            word_vecs = torch.sum(word_vecs, dim=2)
        elif self.mode == "mean":
            # (batch_size, words_num, 1)
            divider = torch.sum(token_ids != 0, dim=-1, keepdim=True)
            word_vecs = torch.sum(word_vecs, dim=2)
            word_vecs = word_vecs / divider
        else:
            word_vecs = torch.max(word_vecs, dim=2)[0]
        return word_vecs
