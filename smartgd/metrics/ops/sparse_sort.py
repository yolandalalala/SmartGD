import torch
from torch import nn


class SparseSort(nn.Module):

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps: float = eps

    def forward(self,
                src: torch.FloatTensor,
                index: torch.LongTensor,
                *,
                dim: int = 0,
                descending: bool = False):
        f_src = src.float()
        f_min, f_max = f_src.min(dim=dim, keepdim=True).values, f_src.max(dim=dim, keepdim=True).values
        norm = (f_src - f_min) / (f_max - f_min + self.eps) + index.float() * (-1) ** int(descending)
        perm = list(torch.meshgrid(*[torch.arange(i).to(src.device) for i in src.shape], indexing='ij'))
        perm[dim] = norm.argsort(dim=dim, descending=descending)

        return src[perm], perm[dim]
