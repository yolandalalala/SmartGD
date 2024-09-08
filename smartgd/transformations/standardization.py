from .base_transformation import BaseTransformation

import torch
from torch_geometric.utils import scatter


class Standardization(BaseTransformation):

    def __init__(self, norm_ord: int = 2):
        super().__init__()
        self.norm_ord: int = norm_ord

    def forward(
            self,
            pos: torch.Tensor,
            apsp: torch.Tensor,
            edge_index: torch.Tensor,
            batch_index: torch.Tensor,
    ) -> torch.Tensor:
        std = torch_scatter.scatter(layout.pos.square(), layout.batch, dim=0, reduce='mean').sqrt()
        normalizer = torch.linalg.norm(std, ord=self.norm_ord, dim=1)
        return layout(layout.pos / normalizer[layout.batch][:, None])
