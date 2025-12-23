import torch

def cross_entropy(logits, targets):
    max_vals, _ = logits.max(dim=-1, keepdim=True)
    # subtract the largest element for numerical stability
    shifted = logits - max_vals
    # get the logits of groundtruth
    target_logits = logits[torch.arange(targets.shape[0]), targets]
    sum_exp = shifted.exp().sum(dim=-1, keepdim=True)
    log_sum_exp = sum_exp.log()
    loss = -((target_logits - max_vals) - log_sum_exp).mean()
    return loss