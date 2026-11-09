import torch
import torch.nn as nn


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        bound = 0.5 / embedding_dim
        self.W_in = nn.Parameter(
            torch.empty((vocab_size, embedding_dim)).uniform_(-bound, bound)
        )
        self.W_out = nn.Parameter(torch.zeros((vocab_size, embedding_dim)))

    def forward(self, c_ids, p_ids, n_ids):
        c_vectors = self.W_in[c_ids]
        p_vectors = self.W_out[p_ids]
        n_vectors = self.W_out[n_ids]

        p_scores = torch.sum(c_vectors * p_vectors, dim=1)
        n_scores = torch.sum(c_vectors.unsqueeze(1) * n_vectors, dim=2)

        return p_scores, n_scores
