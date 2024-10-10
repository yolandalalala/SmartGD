from smartgd.constants import EPS

import torch
from torch import nn
import torch.nn.functional as F


class EdgesIntersect(nn.Module):

    def __init__(self, *, eps: float = EPS):
        super().__init__()
        self.eps = eps

    @staticmethod
    def _cross(v, u):
        return torch.cross(
            F.pad(v, (0, 1)),
            F.pad(u, (0, 1))
        )[:, -1]

    @staticmethod
    def _dot(v, u):
        return (v * u).sum(dim=-1)

    def forward(self, *,
                edge_1_start_pos: torch.Tensor,
                edge_1_end_pos: torch.Tensor,
                edge_2_start_pos: torch.Tensor,
                edge_2_end_pos: torch.Tensor) -> torch.Tensor:

        p, q = edge_1_start_pos, edge_2_start_pos
        r, s = edge_1_end_pos - p, edge_2_end_pos - q

        # shrink by eps
        p += self.eps * r
        q += self.eps * s
        r *= 1 - 2 * self.eps
        s *= 1 - 2 * self.eps

        # get intersection
        qmp = q - p
        qmpxs = self._cross(qmp, s)
        qmpxr = self._cross(qmp, r)
        rxs = self._cross(r, s)
        rdr = self._dot(r, r)
        t = qmpxs / rxs
        u = qmpxr / rxs
        t0 = self._dot(qmp, r) / rdr
        t1 = t0 + self._dot(s, r) / rdr

        # calculate bool
        zero = torch.zeros_like(rxs)
        parallel = rxs.isclose(zero)
        nonparallel = parallel.logical_not()
        collinear = parallel.logical_and(qmpxr.isclose(zero))

        return torch.logical_or(
            collinear.logical_and(
                torch.logical_and(
                    (t0 > 0).logical_or(t1 > 0),
                    (t0 < 1).logical_or(t1 < 1),
                )
            ),
            nonparallel.logical_and(
                torch.logical_and(
                    (0 < t).logical_and(t < 1),
                    (0 < u).logical_and(u < 1),
                )
            )
        )
