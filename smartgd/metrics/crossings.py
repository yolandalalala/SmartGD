from smartgd.constants import EPS

from .ops import EdgesIntersect

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import scatter


class Crossings(nn.Module):

    def __init__(self):
        super().__init__()
        self.intersect = EdgesIntersect(eps=EPS)

    def forward(self, node_pos, edge_index, apsp, batch_index, edge_pair_index) -> torch.Tensor:
        (s1, s2), (e1, e2) = edge_pair_index

        xing = self.intersect(
            edge_1_start_pos=node_pos[s1],
            edge_1_end_pos=node_pos[e1],
            edge_2_start_pos=node_pos[s2],
            edge_2_end_pos=node_pos[e2]
        ).float()

        return scatter(xing, batch_index[s1], reduce="sum")
