from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseTransformation(nn.Module, ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        pos: torch.Tensor,
        apsp: torch.Tensor,
        edge_index: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> torch.Tensor:
        return NotImplemented
