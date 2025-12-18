import torch
import torch.nn as nn
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, weights=None, device=None, dtype=None):
        """

        Args:
            d_model (int): The dimensionality of the RMSNorm input.
            eps: (float): A value added to the denominator for numerical stability.
            weights (Float[Tensor, "d_model"]): RMSNorm weights.
            in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
                dimensions.

        Returns:
            Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
            RMSNorm of the `in_features`.
        """
 
        super().__init__()
        self.eps = eps
        self.d_model = d_model

        if weights is None:
            self.weights = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        else:
            self.weights = nn.Parameter(weights)

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
                dimensions.

        Returns:
            Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
            RMSNorm of the `in_features`.
        """
        in_dtype = in_features.dtype
        in_features = in_features.to(torch.float32)
        # [... d_model] -> [... 1]
        rms = torch.sqrt(1.0/self.d_model * torch.sum(in_features**2, dim=-1, keepdim=True) + self.eps)
 
        # element-wise operation
        return (in_features*self.weights/rms).to(in_dtype)