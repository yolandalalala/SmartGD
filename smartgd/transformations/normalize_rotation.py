from .base_transformation import BaseTransformation

import numpy as np
import torch
from torch_geometric.utils import scatter


class NormalizeRotation(BaseTransformation):

    def __init__(self, base_angle: float = 0):
        super().__init__()
        sin = np.sin(base_angle)
        cos = np.cos(base_angle)
        self.base_rotation: torch.Tensor = torch.tensor(
            [[-sin, +cos],
             [+cos, +sin]]
        ).float()

    def forward(
            self,
            pos: torch.Tensor,
            apsp: torch.Tensor,
            edge_index: torch.Tensor,
            batch_index: torch.Tensor,
    ) -> torch.Tensor:
        outer = torch.einsum('ni,nj->nij', pos, pos)
        cov = scatter(outer, batch_index, dim=0, reduce='mean')
        components = torch.linalg.eigh(cov).eigenvectors
        return torch.einsum(
            'ij,njk,nk->ni',
            self.base_rotation.to(pos.device),
            components[batch_index],
            pos
        ).float()
