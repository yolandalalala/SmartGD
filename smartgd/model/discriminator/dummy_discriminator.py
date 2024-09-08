from smartgd.metrics import Stress

from attrs import define
import torch
from torch import nn


@define(kw_only=True, eq=False, repr=False, slots=False)
class DummyDiscriminator(nn.Module):
    def __attrs_post_init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.stress = Stress()

    def forward(self,
                pos: torch.FloatTensor,
                edge_index: torch.LongTensor,
                edge_attr: torch.FloatTensor,
                batch_index: torch.LongTensor) -> torch.Tensor:
        outputs = self.stress(pos, edge_index, edge_attr[:, 0], batch_index)
        outputs = torch.log(outputs)
        return self.dummy * 0 - outputs
