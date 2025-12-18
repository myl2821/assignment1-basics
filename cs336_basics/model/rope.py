import torch
import torch.nn as nn
from einops import einsum

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self,
                 theta: float,
                 d_k: int,
                 max_seq_len: int,
                 device: torch.device| None = None,
                 dtype: torch.dtype | None = None):
        """
        Args:
            theta (float): RoPE parameter.
            d_k (int): Embedding dimension size for the query or key tensor.
            max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        Returns:
            Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
        """
 
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        positions = torch.arange(max_seq_len, dtype=torch.float, device=device).unsqueeze(1)
        # Float[Tensor, "d_k/2"]
        freqs = torch.arange(0, d_k, 2, device=device) / d_k
        # rotation in complex plane
        inv_freq = 1.0 / (theta**freqs)
        angles = positions * inv_freq

        # pre compute the rotation table, in the shape of Float[Tensor, "d_k/2"]
        self.register_buffer("cos", angles.cos().to(dtype), persistent=False)
        self.register_buffer("sin", angles.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Run RoPE for a given input tensor.

        Args:
            x (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
            token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions

        Returns:
            Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
        """
        cos_pos = self.cos[token_positions]
        sin_pos = self.sin[token_positions]

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        x_rot_even = x_even * cos_pos - x_odd * sin_pos
        x_rot_odd = x_even * sin_pos + x_odd * cos_pos

        d = x_even.shape[-1] * 2
        x = torch.empty(*x_even.shape[:-1], d, device=x_even.device, dtype=x_even.dtype)
        x[..., ::2] = x_rot_even
        x[..., 1::2] = x_rot_odd

        return x
 