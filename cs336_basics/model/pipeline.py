import torch
import math
from .softmax import Softmax
from .linear import Linear
from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int

def scaled_dot_product_attention(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None):
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    att = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    d_k = Q.shape[-1]
    att = att / math.sqrt(d_k)
    if mask != None:
        att = torch.where(mask, att, float("-inf"))

    att = Softmax().forward(att, -1)
    return einsum(att, V, "... queries keys, ... keys d_v -> ... queries d_v")

class CausalMultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None, **kwargs):
        """
        Args:
            d_model (int): Dimensionality of the feedforward input and output.
            num_heads (int): Number of heads to use in multi-headed attention.
        """
 
        assert d_model % num_heads == 0
        super().__init__()

        self.W_q = Linear(d_model, d_model, device, dtype)
        self.W_k = Linear(d_model, d_model, device, dtype)
        self.W_v = Linear(d_model, d_model, device, dtype)
        self.W_o = Linear(d_model, d_model, device, dtype)

        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = d_model // num_heads

    def forward(self, x: Float[torch.Tensor, " ... sequence_length d_in"]):
        """
        Args:
            x (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

        Returns:
            Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
            implementation with the given QKV projection weights and input features.
        """
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape from (..., queries, dim) to (..., heads, queries, head_dim)
        Q = rearrange(Q, "... s (h d) -> ... h s d", h=self.num_heads)
        K = rearrange(K, "... s (h d) -> ... h s d", h=self.num_heads)
        V = rearrange(V, "... s (h d) -> ... h s d", h=self.num_heads)

        queries = x.shape[-2]
        # Create causal mask for self-attention
        mask = ~torch.triu(torch.ones((queries, queries), device=x.device, dtype=torch.bool), diagonal=1)

        MH = scaled_dot_product_attention(Q, K, V, mask)
        MH = rearrange(MH, "... h s d -> ... s (h d)")

        return self.W_o(MH)