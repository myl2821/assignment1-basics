import torch
import torch.nn as nn
import math
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, weight=None, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = weight
        std = math.sqrt(2 / (out_features + in_features))

        if self.weight is None:
            self.weight = nn.Parameter(
                torch.empty(out_features, in_features, device=device, dtype=dtype)
            )
            self._reset_parameters(std)

    def _reset_parameters(self, std):
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        "forward using batch operation, via einsum."
        return einsum(self.weight, input, "d_out d_in, ... d_in -> ... d_out")

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias=False"
