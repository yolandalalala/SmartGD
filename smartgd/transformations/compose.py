from .base_transformation import BaseTransformation

import torch
from torch import nn


class Compose(BaseTransformation):

    def __init__(self, *transformations: BaseTransformation):
        super().__init__()
        self.transformations: nn.ModuleList[BaseTransformation] = nn.ModuleList(transformations)

    def forward(
            self,
            pos: torch.Tensor,
            apsp: torch.Tensor,
            edge_index: torch.Tensor,
            batch_index: torch.Tensor,
    ) -> torch.Tensor:
        for transformation in self.transformations:
            pos = transformation(pos, apsp, edge_index, batch_index)
        return pos
