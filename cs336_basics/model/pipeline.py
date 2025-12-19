import torch
import math
from .softmax import Softmax
from .linear import Linear
from .rope import RotaryPositionalEmbedding
from .embedding import Embedding
from .norm import RMSNorm
from .swiglu import SwigLU
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

    def forward(
            self,
            x: Float[torch.Tensor, " ... sequence_length d_in"],
            rope: RotaryPositionalEmbedding | None = None,
            token_positions: torch.Tensor | None = None):
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

        if rope is not None:
            if token_positions is None:
                token_positions = torch.arange(queries, dtype=torch.float, device=Q.device).long()
            Q = rope.forward(Q, token_positions)
            K = rope.forward(K, token_positions)

        # Create causal mask for self-attention
        mask = ~torch.triu(torch.ones((queries, queries), device=x.device, dtype=torch.bool), diagonal=1)

        MH = scaled_dot_product_attention(Q, K, V, mask)
        MH = rearrange(MH, "... h s d -> ... s (h d)")

        return self.W_o(MH)

class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        w1_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        w3_weight: torch.Tensor,
        rope: RotaryPositionalEmbedding | None = None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        """
        Args:
            d_model (int): The dimensionality of the Transformer block input.
            num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
                evenly divisible by `num_heads`.
            d_ff (int): Dimensionality of the feed-forward inner layer.
            max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
            theta (float): RoPE parameter.
            w1_weight: Weight of the first linear transformation in the FFN.
            w2_weight: Weight of the first linear transformation in the FFN.
            w3_weight: Weight of the first linear transformation in the FFN.
        """
        super().__init__()

        self.rope = rope

        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, device, dtype, **kwargs)

        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwigLU(d_model, d_ff, w1_weight, w2_weight, w3_weight, device, dtype)

    def forward(self, x: torch.Tensor):
        """
        Given the weights of a pre-norm Transformer block and input features,
        return the output of running the Transformer block on the input features.

        This function should use RoPE.
        Depending on your implementation, you may simply need to pass the relevant args
        to your TransformerBlock constructor, or you may need to initialize your own RoPE
        class and pass that instead.

           in_features (Float[Tensor, "batch sequence_length d_model"]):
                Tensor to run your implementation on.

        Returns:
            Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
            running the Transformer block on the input features while using RoPE.
        """

        x = x + self.attn(self.ln1(x), self.rope)
        x = x + self.ffn(self.ln2(x))
        return x