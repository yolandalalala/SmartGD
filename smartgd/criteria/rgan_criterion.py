from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class RGANCriterion(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, encourage: torch.Tensor, discourage: torch.Tensor) -> torch.Tensor:
        return torch.mean(- F.logsigmoid(encourage - discourage))
