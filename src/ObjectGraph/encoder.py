"""Graph encoder backbone (GraphSAGE by default).

Every constructor default reproduces the original two-layer un-regularised SAGE exactly, and
the first two convolutions keep the attribute names ``c1``/``c2`` so shipped checkpoints load
unchanged. Sweep axes (depth, dropout, backbone, edge weighting) are opt-in.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class WeightedSAGEConv(nn.Module):
    """SAGEConv with a weighted neighbourhood mean.

    ``SAGEConv`` silently ignores ``edge_weight``, so the paper's Eq. (2) weighting
    w_ab = c_ab / sqrt(c_a * c_b) cannot be expressed with it. This reimplements the same
    computation -- out = lin_l(mean_j x_j) + lin_r(x) -- with the mean replaced by a
    weighted mean. Parameter names match ``SAGEConv`` so state dicts stay interchangeable,
    and at uniform weights the two modules are numerically identical (verified).
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.lin_l = nn.Linear(in_dim, out_dim, bias=bias)
        self.lin_r = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, edge_index, edge_weight=None):
        src, dst = edge_index[0], edge_index[1]
        w = torch.ones(src.numel(), device=x.device, dtype=x.dtype) if edge_weight is None else edge_weight
        num = torch.zeros_like(x).index_add_(0, dst, x[src] * w.unsqueeze(1))
        den = torch.zeros(x.size(0), device=x.device, dtype=x.dtype).index_add_(0, dst, w)
        agg = num / den.clamp_min(1e-12).unsqueeze(1)
        return self.lin_l(agg) + self.lin_r(x)


def _make_conv(backbone: str, in_dim: int, out_dim: int, heads: int, aggr: str = "mean"):
    if backbone == "sage":
        # aggr='mean' is SAGEConv's own default, so passing it explicitly leaves the published
        # configuration bit-identical while opening GraphSAGE's aggregator ablation axis.
        return SAGEConv(in_dim, out_dim, aggr=aggr)
    if backbone == "wsage":
        return WeightedSAGEConv(in_dim, out_dim)
    if backbone == "gat":
        # concat=False keeps the output dim at hidden_dim, so backbones are compared at
        # matched representation size rather than matched head count.
        return GATConv(in_dim, out_dim, heads=heads, concat=False)
    if backbone == "gcn":
        return GCNConv(in_dim, out_dim)
    raise ValueError(f"unknown backbone: {backbone}")


class GraphEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        normalize: bool = True,
        num_layers: int = 2,
        dropout: float = 0.0,
        backbone: str = "sage",
        heads: int = 4,
        aggr: str = "mean",
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.normalize = normalize
        self.dropout = dropout
        self.backbone = backbone
        self.num_layers = num_layers
        # Only 'sage' has a selectable aggregator; the others fix their own. Rejecting the
        # combination rather than silently ignoring it means an arm cannot quietly become the
        # control -- the failure mode --item_prop and --freeze_fusion both needed guards for.
        if aggr != "mean" and backbone != "sage":
            raise ValueError(f"aggr={aggr!r} is only meaningful for backbone='sage', not {backbone!r}")
        self.aggr = aggr
        convs = []
        for i in range(num_layers):
            conv = _make_conv(backbone, in_dim if i == 0 else hidden_dim, hidden_dim, heads,
                              aggr=aggr)
            # c1/c2 are the historical attribute names; keeping them means the shipped
            # checkpoints still load into the default configuration.
            setattr(self, f"c{i + 1}", conv)
            convs.append(conv)
        self.convs = convs  # plain list: the modules are already registered by setattr

    def _apply_conv(self, conv, x, edge_index, edge_weight):
        if isinstance(conv, (WeightedSAGEConv, GCNConv)):
            return conv(x, edge_index, edge_weight)
        return conv(x, edge_index)

    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs):
            x = self._apply_conv(conv, x, edge_index, edge_weight)
            if i < len(self.convs) - 1:  # no activation or dropout after the final layer
                x = F.relu(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return F.normalize(x, p=2, dim=1) if self.normalize else x

    def load_graphsage_state(self, state: dict):
        self.load_state_dict({k: v for k, v in state.items() if k in self.state_dict()}, strict=False)
