from smartgd.constants import EPS
from ..common import EdgeFeatureExpansion, NNConvLayer
from .discriminator_block import DiscriminatorBlock

from dataclasses import dataclass, field

from attrs import define
import torch
from torch import nn
import torch_geometric as pyg


@define(kw_only=True, eq=False, repr=False, slots=False)
class Discriminator(nn.Module):
    @dataclass(kw_only=True, frozen=True)
    class Params:
        num_layers: int
        hidden_width: int
        edge_net_shared_depth: int
        edge_net_embedded_depth: int
        edge_net_width: int
        edge_attr_dim: int

    @dataclass(kw_only=True, frozen=True)
    class Config:
        pooling: str | list[str] = field(default_factory=lambda: ["sum", "mean", "max"])

    params: Params = Params(
        num_layers=9,
        hidden_width=16,
        edge_net_shared_depth=8,
        edge_net_embedded_depth=8,
        edge_net_width=64,
        edge_attr_dim=2
    )
    config: Config = Config()
    edge_net_config: DiscriminatorBlock.EdgeNetConfig = DiscriminatorBlock.EdgeNetConfig()
    gnn_config: NNConvLayer.NNConvConfig = NNConvLayer.NNConvConfig(
        aggr="add",
        residual=True
    )
    edge_feat_expansion: EdgeFeatureExpansion.Expansions = EdgeFeatureExpansion.Expansions(
        src_feat=True,
        dst_feat=True,
        diff_vec=False,
        unit_vec=False,
        vec_norm=False,
        vec_norm_inv=False,
        vec_norm_square=False,
        vec_norm_inv_square=False,
        edge_attr_inv=False,
        edge_attr_square=False,
        edge_attr_inv_square=False
    )
    eps: float = EPS

    def __post_init__(self):
        super().__init__()

        self.block: DiscriminatorBlock = DiscriminatorBlock(
            params=DiscriminatorBlock.Params(
                in_dim=2,
                out_dim=self.params.hidden_width,
                hidden_width=self.params.hidden_width,
                hidden_depth=self.params.num_layers,
                edge_attr_dim=self.params.edge_attr_dim,
                node_attr_dim=2
            ),
            edge_net_params=DiscriminatorBlock.EdgeNetParams(
                shared_depth=self.params.edge_net_shared_depth,
                embedded_depth=self.params.edge_net_embedded_depth,
                hidden_width=self.params.edge_net_width
            ),
            edge_net_config=self.edge_net_config,
            gnn_config=self.gnn_config,
            edge_feat_expansion=self.edge_feat_expansion,
            eps=self.eps
        )

        self.readout: pyg.nn.Aggregation = pyg.nn.MultiAggregation(
            aggrs=self.config.pooling,  # TODO: deal with single str or list with a single element
            mode="proj",
            mode_kwargs=dict(
                in_channels=self.params.hidden_width,
                out_channels=1
            )
        )

    def forward(self,
                pos: torch.FloatTensor,
                edge_index: torch.LongTensor,
                edge_attr: torch.FloatTensor,
                batch_index: torch.LongTensor) -> torch.Tensor:
        node_feat = self.block(
            node_feat=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch_index=batch_index
        )
        outputs = self.readout(node_feat, batch_index)
        return outputs.flatten()
