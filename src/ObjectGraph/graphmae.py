"""GraphMAE: masked feature reconstruction for graph encoder fine-tuning (Hou et al., KDD 2022)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import GraphEncoder


def sce_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 3.0) -> torch.Tensor:
    pred = F.normalize(pred, p=2, dim=-1)
    target = F.normalize(target, p=2, dim=-1)
    return (1 - (pred * target).sum(dim=-1)).pow_(alpha).mean()


class GraphMAE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        mask_rate: float = 0.75,
        alpha: float = 3.0,
        replace_rate: float = 0.1,
        num_layers: int = 2,
        dropout: float = 0.0,
        backbone: str = "sage",
        heads: int = 4,
        remask: bool = False,
    ):
        super().__init__()
        self.mask_rate = mask_rate
        self.alpha = alpha
        self.replace_rate = replace_rate
        self.remask = remask
        # The trailing four arguments default to the original fixed two-layer SAGE, so
        # train_mae()/train_full() are unaffected; they exist so the stage ablation can run
        # stage 2 on the same architecture as the stage-1 arm it is being compared against.
        self.encoder = GraphEncoder(
            in_dim, hidden_dim, normalize=False,
            num_layers=num_layers, dropout=dropout, backbone=backbone, heads=heads,
        )
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        self.enc_to_dec = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.decoder = nn.Linear(hidden_dim, in_dim)

    def _mask(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = x.size(0)
        perm = torch.randperm(n, device=x.device)
        n_mask = max(1, int(self.mask_rate * n))
        mask_nodes = perm[:n_mask]
        out = x.clone()
        if self.replace_rate > 0:
            n_rep = int(self.replace_rate * n_mask)
            noise_idx = torch.randperm(n, device=x.device)[:n_rep]
            out[mask_nodes[:n_mask - n_rep]] = 0.0
            out[mask_nodes[:n_mask - n_rep]] += self.enc_mask_token
            out[mask_nodes[n_mask - n_rep :]] = x[noise_idx]
        else:
            out[mask_nodes] = 0.0
            out[mask_nodes] += self.enc_mask_token
        return out, mask_nodes

    def forward(self, x, edge_index, edge_weight=None) -> torch.Tensor:
        x_in, mask_nodes = self._mask(x)
        h = self.encoder(x_in, edge_index, edge_weight)
        h = self.enc_to_dec(h)
        if self.remask:
            # GraphMAE's re-mask: zero the masked rows again before decoding, so the decoder
            # must recover them from their neighbours. That only works with a *GNN* decoder.
            # This decoder is nn.Linear, so decoder(0) is just its bias and the loss becomes
            # a constant w.r.t. the encoder -- gradient to every encoder parameter is exactly
            # zero (verified: only decoder.bias receives gradient). Kept behind a flag,
            # defaulting off, so the shipped behaviour stays reproducible for the audit.
            h = h.clone()
            h[mask_nodes] = 0.0
        return sce_loss(self.decoder(h)[mask_nodes], x[mask_nodes], self.alpha)

    @torch.no_grad()
    def embed(self, x, edge_index, edge_weight=None) -> torch.Tensor:
        # Identical to the previous hand-rolled relu(c1) -> c2 -> normalize at two layers,
        # but follows the encoder's own forward so depth and backbone are not hard-coded.
        return F.normalize(self.encoder(x, edge_index, edge_weight), p=2, dim=1)

    def load_encoder_checkpoint(self, state: dict):
        self.encoder.load_graphsage_state(state)
