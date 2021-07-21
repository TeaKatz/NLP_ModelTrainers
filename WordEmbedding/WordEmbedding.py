from torch.nn import Module, Embedding


class WordEmbedding(Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=0):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        self.l_embedding = Embedding(num_embeddings=num_embeddings, 
                                     embedding_dim=embedding_dim, 
                                     padding_idx=padding_idx)

    def forward(self, token_ids):
        """
        token_ids: (batch_size, words_num)

        Return
        word_vecs: (batch_size, words_num, embedding_dim)
        """
        word_vecs = self.l_embedding(token_ids)
        return word_vecs
