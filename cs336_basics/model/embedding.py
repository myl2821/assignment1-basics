import torch
import torch.nn as nn
import math
from einops import einsum

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, weights=None, device=None, dtype=None):
        """
        Args:
            num_embeddings (int): The number of embeddings in the vocabulary
            embedding_dim (int): The size of the embedding dimension, a.k.a dim_model
            weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        """


        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        if self.weight is None:
            self.weight = nn.Parameter(
                torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
            )
            self._reset_parameters()
        else:
            self.weight = nn.Parameter(weights)


    def _reset_parameters(self):
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

        Args:
            token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

        Returns:
            Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
        """

        return self.weight[token_ids]