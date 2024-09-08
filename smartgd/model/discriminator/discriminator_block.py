from smartgd.constants import EPS
from ..common import NNConvLayer, NNConvBasicLayer, EdgeFeatureExpansion
from .discriminator_edge_net import DiscriminatorEdgeNet

from attrs import define
from typing import Optional
from dataclasses import dataclass

import torch
from torch import nn


@define(kw_only=True, eq=False, repr=False, slots=False)
class DiscriminatorBlock(nn.Module):

    @dataclass(kw_only=True, frozen=True)
    class Params:
        in_dim: int
        out_dim: int
        hidden_width: int
        hidden_depth: int
        edge_attr_dim: int
        node_attr_dim: int

    @dataclass(kw_only=True, frozen=True)
    class EdgeNetParams:
        shared_depth: int  # TODO: depth=0
        embedded_depth: int  # TODO: depth=0
        hidden_width: int

    @dataclass(kw_only=True, frozen=True)
    class EdgeNetConfig:
        hidden_act: str = "leaky_relu"
        out_act: Optional[str] = None
        bn: Optional[str] = "batch_norm"
        dp: float = 0.0
        residual: bool = True

    params: Params
    edge_net_params: EdgeNetParams
    edge_net_config: EdgeNetConfig = EdgeNetConfig()
    gnn_config: NNConvLayer.NNConvConfig = NNConvLayer.NNConvConfig(
        aggr="add",
        residual=True
    )
    edge_feat_expansion: EdgeFeatureExpansion.Expansions = EdgeFeatureExpansion.Expansions()
    eps: float = EPS

    def __attrs_post_init__(self):
        super().__init__()

        self.shared_edge_net: DiscriminatorEdgeNet = DiscriminatorEdgeNet(
            params=DiscriminatorEdgeNet.Params(
                edge_attr_dim=self.params.edge_attr_dim,
                node_feat_dim=self.params.node_attr_dim,
                out_dim=self.edge_net_params.hidden_width,
            ),
            config=DiscriminatorEdgeNet.Config(
                hidden_depth=self.edge_net_params.shared_depth,
                hidden_width=self.edge_net_params.hidden_width,
                out_act=self.edge_net_config.hidden_act,
                hidden_act=self.edge_net_config.hidden_act,
                bn=self.edge_net_config.bn,
                dp=self.edge_net_config.dp,
                residual=self.edge_net_config.residual
            ),
            edge_feat_expansion=self.edge_feat_expansion,
            eps=self.eps
        )

        self.layer_list: nn.ModuleList[DiscriminatorBlock] = nn.ModuleList()

        in_dims = [self.params.in_dim] + [self.params.hidden_width] * self.params.hidden_depth
        out_dims = [self.params.hidden_width] * self.params.hidden_depth + [self.params.out_dim]
        for layer_index, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            self.layer_list.append(NNConvLayer(
                layer_index=-1,
                params=NNConvBasicLayer.Params(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    edge_feat_dim=self.edge_net_params.hidden_width
                ),
                nnconv_config=NNConvLayer.NNConvConfig(
                    dense=self.gnn_config.dense,
                    bn=self.gnn_config.bn,
                    act=self.gnn_config.act,
                    dp=self.gnn_config.dp,
                    residual=self.gnn_config.residual,
                    aggr=self.gnn_config.aggr,
                    root_weight=self.gnn_config.root_weight
                ),
                edge_net_config=NNConvLayer.EdgeNetConfig(
                    hidden_dims=[self.edge_net_params.hidden_width] * self.edge_net_params.embedded_depth,
                    hidden_act=self.edge_net_config.hidden_act,
                    out_act=self.edge_net_config.out_act,
                    bn=self.edge_net_config.bn,
                    dp=self.edge_net_config.dp,
                    residual=self.edge_net_config.residual
                )
            ))

    def forward(self, *,
                node_feat: torch.FloatTensor,
                edge_index: torch.LongTensor,
                edge_attr: torch.FloatTensor,
                batch_index: torch.LongTensor) -> torch.FloatTensor:
        edge_feat = self.shared_edge_net(
            node_feat=node_feat,
            edge_attr=edge_attr,
            edge_index=edge_index
        )
        outputs = torch.ones_like(node_feat)
        for layer in self.layer_list:
            outputs, _, _ = layer(
                node_feat=outputs,
                edge_feat=edge_feat,
                edge_index=edge_index,
                batch_index=batch_index,
                num_sampled_nodes_per_hop=None,
                num_sampled_edges_per_hop=None,
            )
        return outputs
