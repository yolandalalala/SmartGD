from typing import Optional

import torch
from torch import nn


class Reduce(nn.Module):

    reduce: str
    ignore_nan: bool

    def __init__(self, method: Optional[str] = None, ignore_nan: bool = True):
        super().__init__()
        self.reduce = method or "none"
        self.ignore_nan = ignore_nan

    def forward(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        if self.reduce == "none":
            return x
        if self.ignore_nan:
            x = x[~x.isnan()]
        if self.reduce == "mean":
            return torch.mean(x, dim)
        if self.reduce == "sum":
            return torch.sum(x, dim)
        if self.reduce == "min":
            return torch.min(x, dim)[0]
        if self.reduce == "max":
            return torch.max(x, dim)[0]
        assert False, f"Unknown reduce type {self.reduce}!"
