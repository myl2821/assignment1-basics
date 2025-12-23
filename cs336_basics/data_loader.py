import numpy as np
import torch

def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str):
    """
    Takes a numpy array of token IDs, batches them, and returns a pair of tensors:
    - (batch_size, context_length) — the actual batch
    - (batch_size, context_length) — the next token ID for each sample in the batch

    Args:
      x: np.ndarray — integer array of training data
      batch_size: int — number of samples per batch
      context_length: int — length of each sequence in batch
      device: str — device to load the data on
    """

    assert x.ndim == 1, "x must be a 1D array of token IDs"
    assert len(x) > context_length, "x is too short for the given context_length"

    # Maximum valid starting index to ensure we have enough tokens for a full sequence plus one
    max_start_idx = len(x) - context_length - 1

    start_indices = torch.randint(
        0, max_start_idx + 1, (batch_size,), device=device
    )

    # Build batch
    idx = start_indices.unsqueeze(1) + torch.arange(
        context_length, device=device
    )

    inputs = torch.from_numpy(x[idx])
    targets = torch.from_numpy(x[idx+1])
    
    if device.startswith("cuda"):
        # Pin memory if on GPU
        inputs, targets = (
            inputs.pin_memory().to(device, non_blocking=True),
            targets.pin_memory().to(device, non_blocking=True),
        )
    else:
        inputs, targets = inputs.to(device), targets.to(device)

    return inputs, targets