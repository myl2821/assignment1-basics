import torch
import torch.nn as nn
from einops import einsum

class SwigLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, w1_weight: torch.Tensor, w2_weight: torch.Tensor, w3_weight: torch.Tensor, device=None, dtype=None):
        """

        Args:
            d_model (int): Dimensionality of the feedforward input and output.
            d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
            w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
            w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
            w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        """
 
        super().__init__()
        self.d_model = d_model
        self.w1_weight = nn.Parameter(w1_weight)
        self.w2_weight = nn.Parameter(w2_weight)
        self.w3_weight = nn.Parameter(w3_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Given the weights of a SwiGLU network, return
        the output of your implementation with these weights.

        Args:
            x (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

        Returns:
            Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
        """

        w1x = einsum(self.w1_weight, x, "d_ff d_model, ... d_model -> ... d_ff")
        silu = w1x * torch.sigmoid(w1x)
        w3x = einsum(self.w3_weight, x, "d_ff d_model, ... d_model -> ... d_ff")
        swiglu = einsum(self.w2_weight, silu*w3x, "d_model d_ff, ... d_ff -> ... d_model")
        return swiglu
 