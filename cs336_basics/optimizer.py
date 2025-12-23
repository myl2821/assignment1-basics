from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            for param in group["params"]:
                if param.grad is None:
                    continue

                state = self.state[param]
                t = state.get("t", 0)
                grad = param.grad
                param.data -= lr / math.sqrt(t+1) * grad
                state["t"] = t + 1

        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self,
                 params: Iterable[torch.nn.Parameter],
                 lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.01,
                 ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]

                m = state["m"]
                v = state["v"]

                # 1. update moments
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 2. calculate learning rate
                lr_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)

                # 3. update parameters
                p.data.addcdiv_(m, v.sqrt().add_(eps), value=-lr_t)

                # 4. weight decay
                if wd != 0:
                    p.data.add_(p.data, alpha=-lr * wd)

        return loss

                
def get_lr_cosine_schedule(
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> Callable[[int], float]:
    """
    Returns a learning rate schedule function that implements a cosine learning rate schedule with warmup.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): The maximum learning rate.
        min_learning_rate (float): The minimum learning rate.
        warmup_iters (int): The number of iterations to linearly increase the learning rate.
        cosine_cycle_iters (int): The number of iterations for one cosine cycle.

    Returns:
        A function that takes the current iteration (int) and returns the learning rate (float).
    """
    def lr_fn(it: int) -> float:
        if it < warmup_iters:
            return max_learning_rate * (it / warmup_iters)
        if it > cosine_cycle_iters:
            return min_learning_rate

        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine_decay

    return lr_fn

def gradiant_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float,
    norm_type: float = 2.0,
) -> torch.Tensor:
    """
    Clips the gradients of the given parameters to have a maximum norm of `max_norm`.

    Args:
        parameters (Iterable[torch.nn.Parameter]): The parameters whose gradients will be clipped.
        max_norm (float): The maximum allowed norm of the gradients.
        norm_type (float): The type of the used p-norm. Can be 'inf' for infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type

    total_norm = total_norm ** (1. / norm_type)

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)

    return total_norm