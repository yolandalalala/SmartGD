from .base_transformation import BaseTransformation

import torch
from torch_geometric.utils import scatter


class Center(BaseTransformation):

    def __init__(self):
        super().__init__()

    def forward(
            self,
            pos: torch.Tensor,
            apsp: torch.Tensor,
            edge_index: torch.Tensor,
            batch_index: torch.Tensor,
    ) -> torch.Tensor:
        center = scatter(pos, batch_index, dim=0, reduce='mean')
        return pos - center[batch_index]
