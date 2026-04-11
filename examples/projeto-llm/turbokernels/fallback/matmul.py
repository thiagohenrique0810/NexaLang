"""CPU fallback matmul — plain PyTorch."""

import torch


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a, b)


def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    out = torch.matmul(x, weight.t())
    if bias is not None:
        out += bias
    return out
