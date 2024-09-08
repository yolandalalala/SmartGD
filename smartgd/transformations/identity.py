from .base_transformation import BaseTransformation

import torch


class Identity(BaseTransformation):

    def __init__(self):
        super().__init__()

    def forward(
            self,
            pos: torch.Tensor,
            apsp: torch.Tensor,
            edge_index: torch.Tensor,
            batch_index: torch.Tensor,
    ) -> torch.Tensor:
        return pos
