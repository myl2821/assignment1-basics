import torch
import torch.nn as nn

class Softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Given a tensor of inputs, return the output of softmaxing the given `dim`
        of the input.

        Args:
            x (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
            dim (int): Dimension of the `in_features` to apply softmax to.

        Returns:
            Float[Tensor, "..."]: Tensor of with the same shape as `x` with the output of
            softmax normalizing the specified `dim`.
        """
        x_max = x.max(dim=dim, keepdim=True).values
        x_exp = torch.exp(x - x_max)
        return x_exp / x_exp.sum(dim=dim, keepdim=True)
 