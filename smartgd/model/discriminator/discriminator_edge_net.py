from smartgd.constants import EPS
from ..common import MLP, EdgeFeatureExpansion

from typing import Optional
from dataclasses import dataclass

from attrs import define
import torch
from torch import nn


@define(kw_only=True, eq=False, repr=False, slots=False)
class DiscriminatorEdgeNet(nn.Module):

    @dataclass(kw_only=True, frozen=True)
    class Params:
        edge_attr_dim: int
        node_feat_dim: int
        out_dim: int

    @dataclass(kw_only=True, frozen=True)
    class Config:
        hidden_depth: int = 0
        hidden_width: Optional[int] = None
        out_act: Optional[str] = None
        hidden_act: str = "leaky_relu"
        bn: Optional[str] = "batch_norm"
        dp: float = 0.0
        residual: bool = True

    params: Params
    config: Config = Config()
    edge_feat_expansion: EdgeFeatureExpansion.Expansions = EdgeFeatureExpansion.Expansions()
    eps: float = EPS

    def __post_init__(self):
        super().__init__()

        self.edge_feature_provider: EdgeFeatureExpansion = EdgeFeatureExpansion(
            config=EdgeFeatureExpansion.Config(
                node_feat_dim=self.params.node_feat_dim,
                edge_attr_dim=self.params.edge_attr_dim,
            ),
            expansions=self.edge_feat_expansion,
            eps=self.eps
        )

        self.mlp: MLP = MLP(
            params=MLP.Params(
                in_dim=self.edge_feature_provider.get_feature_channels(),
                out_dim=self.params.out_dim,
                hidden_dims=[self.config.hidden_width] * self.config.hidden_depth
            ),
            config=MLP.Config(
                hidden_act=self.config.hidden_act,
                out_act=self.config.out_act,
                bn=self.config.bn,
                dp=self.config.dp,
                residual=self.config.residual
            )
        )

    def forward(self, *,
                node_feat: torch.FloatTensor,
                edge_attr: torch.FloatTensor,
                edge_index: torch.LongTensor) -> torch.FloatTensor:
        outputs = self.edge_feature_provider(
            node_feat=node_feat,
            edge_attr=edge_attr,
            edge_index=edge_index
        )
        outputs = self.mlp(outputs)
        return outputs
