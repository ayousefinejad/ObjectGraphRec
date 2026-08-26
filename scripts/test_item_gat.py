#!/usr/bin/env python3
"""Verify the GAT item-graph propagation (--item_prop gat).

    python scripts/test_item_gat.py

Three things have to hold before this arm is worth 3 GPU-hours:

1. It reduces to the published operator at reduce-init, so any measured difference is
   attributable to learned attention rather than to a re-initialised model.
2. It preserves the gradient to the fusion parameters. `item_adj @ h` is their ONLY path to
   the loss; a conv that consumes only edge_index severs it and makes the whole modality
   fusion a silent no-op. GraphMAE stage 2 was exactly this bug and went unnoticed.
3. The gradient check can actually FAIL. A negative control (edge weight dropped from the
   message) must be rejected, otherwise check 2 proves nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Models.py parses argv at import time.
_REAL_ARGV = sys.argv
sys.argv = [_REAL_ARGV[0]]

import torch                                                        # noqa: E402
import torch.nn.functional as F                                     # noqa: E402

from Models import ItemGATConv, _segment_softmax                    # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  ok   " if cond else "  FAIL ") + name + (f"  {detail}" if detail else ""))


def edges_of(A):
    """item_adj convention: entry [i, j] is the weight of j's contribution to i.

    So row = dst, col = src. dense_to_sparse() returns the transpose of this and would
    silently propagate A.t() instead --- item_adj is asymmetric whenever knn_sym='none',
    which is the default.
    """
    dst, src = A.nonzero(as_tuple=True)
    return src, dst, A[dst, src]


def main():
    torch.manual_seed(0)
    n, d = 40, 8

    # asymmetric graph: 4 random neighbours per row, so A != A.t()
    A = torch.zeros(n, n)
    for i in range(n):
        j = torch.randperm(n)[:4]
        A[i, j] = torch.rand(4) + 0.1
    x = torch.randn(n, d)
    src, dst, w = edges_of(A)

    print("1. reduction to the published operator")
    g = ItemGATConv(d, heads=2, init='reduce')
    with torch.no_grad():
        got = F.normalize(g(x, src, dst, w, n), p=2, dim=1)
    want = F.normalize(A @ x, p=2, dim=1)
    err = (got - want).abs().max().item()
    check("reduce-init == item_adj @ h (after F.normalize)", err < 1e-5, f"max|d|={err:.2e}")

    print("2. propagation direction")
    wrong = F.normalize(A.t() @ x, p=2, dim=1)
    check("does NOT match A.t() @ h (dense_to_sparse transposition trap)",
          (got - wrong).abs().max().item() > 1e-3,
          f"max|d|={(got - wrong).abs().max().item():.2e}")

    print("3. gradient reaches the fusion parameters")
    theta = torch.zeros(3, requires_grad=True)                 # stand-in for modal_weight
    mats = [torch.rand(n, n) * (A > 0) for _ in range(3)]
    fused = sum(m * ww for m, ww in zip(mats, torch.softmax(theta, 0)))
    s, dd, ww = edges_of(fused.detach())                       # indices only, no grad
    ww = fused[dd, s]                                          # differentiable gather
    g2 = ItemGATConv(d, heads=2, init='reduce')
    g2(x, s, dd, ww, n).sum().backward()
    ok = theta.grad is not None and theta.grad.abs().sum() > 0 and torch.isfinite(theta.grad).all()
    check("modal_weight stand-in receives a non-zero finite gradient", bool(ok),
          f"grad={None if theta.grad is None else theta.grad.tolist()}")

    print("4. negative control -- the check above must be able to fail")
    class Naive(ItemGATConv):
        """Drops the edge weight from the message, i.e. the edge_index-only implementation."""
        def forward(self, x, src, dst, w, n):
            return super().forward(x, src, dst, torch.ones_like(w), n)
    theta2 = torch.zeros(3, requires_grad=True)
    fused2 = sum(m * ww2 for m, ww2 in zip(mats, torch.softmax(theta2, 0)))
    s2, d2, _ = edges_of(fused2.detach())
    w2 = fused2[d2, s2]
    Naive(d, heads=2, init='reduce')(x, s2, d2, w2, n).sum().backward()
    severed = theta2.grad is None or theta2.grad.abs().sum() == 0
    check("edge_index-only variant severs the gradient (control fails as it must)",
          bool(severed), f"grad={None if theta2.grad is None else theta2.grad.tolist()}")

    print("5. isolated rows and stacking")
    A2 = A.clone()
    A2[3] = 0                                                  # node 3 has no in-edges
    s3, d3, w3 = edges_of(A2)
    with torch.no_grad():
        out = ItemGATConv(d, heads=2, init='reduce')(x, s3, d3, w3, n)
    check("zero-in-degree row stays exactly zero and finite",
          bool((out[3].abs().sum() == 0) and torch.isfinite(out).all()))
    check("F.normalize of that row is zero, not NaN",
          bool(torch.isfinite(F.normalize(out, p=2, dim=1)).all()))
    h = x
    for _ in range(2):
        with torch.no_grad():
            h = ItemGATConv(d, heads=2, init='reduce')(h, src, dst, w, n)
    check("n_layers=2 stacking stays finite", bool(torch.isfinite(h).all()))

    print("6. segment softmax")
    lg = torch.randn(len(dst), 2)
    al = _segment_softmax(lg, dst, n)
    sums = torch.zeros(n, 2).index_add_(0, dst, al)
    present = torch.zeros(n).index_add_(0, dst, torch.ones(len(dst))) > 0
    check("rows with in-edges sum to 1", bool((sums[present] - 1).abs().max() < 1e-5))
    check("xavier init differs from reduce init",
          not torch.allclose(ItemGATConv(d, heads=2, init='xavier').lin.weight,
                             torch.eye(d), atol=1e-6))

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
