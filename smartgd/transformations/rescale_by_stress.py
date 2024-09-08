from .base_transformation import BaseTransformation

import torch
from torch_geometric.utils import scatter


class RescaleByStress(BaseTransformation):

    def __init__(self):
        super().__init__()

    def forward(
            self,
            pos: torch.Tensor,
            apsp: torch.Tensor,
            edge_index: torch.Tensor,
            batch_index: torch.Tensor,
    ) -> torch.Tensor:
        src_pos, dst_pos = pos[edge_index[0]], pos[edge_index[1]]
        dist = (dst_pos - src_pos).norm(dim=1)
        u_over_d = dist / apsp
        scatterd_u_over_d_2 = scatter(u_over_d ** 2, batch_index[edge_index[0]])
        scatterd_u_over_d = scatter(u_over_d, batch_index[edge_index[0]])
        scale = scatterd_u_over_d_2 / scatterd_u_over_d
        return pos / scale[batch_index][:, None]
