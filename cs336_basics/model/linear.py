import torch
import torch.nn as nn
import math
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, weights=None, device=None, dtype=None):
        """
        Given the weights of a Linear layer, compute the transformation of a batched input.

        Args:
            in_features (int): The size of the input dimension
            out_features (int): The size of the output dimension
            weights (Float[Tensor, "d_out d_in"]): The linear weights to use
            device(torch.device): Device to store the parameters on
            dtype(torch.dtype): Data type of the parameters
        """

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = weights
        std = math.sqrt(2 / (out_features + in_features))

        if self.weight is None:
            self.weight = nn.Parameter(
                torch.empty(out_features, in_features, device=device, dtype=dtype)
            )
            self._reset_parameters(std)

    def _reset_parameters(self, std):
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Given the weights of a Linear layer, compute the transformation of a batched input.

        Args:
            input (Float[Tensor, "... d_in"]): The output tensor to apply the function to

        Returns:
            Float[Tensor, "... d_out"]: The transformed output of your linear module.
        """

        return einsum(self.weight, input, "d_out d_in, ... d_in -> ... d_out")

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias=False"
