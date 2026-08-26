#!/usr/bin/env python3
"""Verify the tie-aware kNN variants, and that the default path is bit-identical.

    python scripts/test_knn_variants.py            # ops + tiny fixtures
    python scripts/test_knn_variants.py --real     # also check the real 841 MB cached tensors

The point of every check here is that a knob either changes the graph in the way it claims to or
changes nothing at all. A silently-ignored flag and a silently-stale cache both look like a null
result downstream, which is exactly the conclusion this study is trying to reach honestly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Models.py parses argv at import time.
_REAL_ARGV = sys.argv
sys.argv = [_REAL_ARGV[0]]

import torch                                                        # noqa: E402

import Models                                                       # noqa: E402
from Models import (build_knn_neighbourhood, build_sim, compute_normalized_laplacian,  # noqa: E402
                    topk_boundary_gap, symmetrise, _topk_from_scores, _NEG)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  ok   " if cond else "  FAIL ") + name + (f"  {detail}" if detail else ""))


class Cfg:
    """Temporarily set Models.args fields; Models reads its flags off that module global."""

    def __init__(self, **kw):
        self.kw = kw

    def __enter__(self):
        self.old = {k: getattr(Models.args, k) for k in self.kw}
        for k, v in self.kw.items():
            setattr(Models.args, k, v)

    def __exit__(self, *a):
        for k, v in self.old.items():
            setattr(Models.args, k, v)


class Stub:
    """Just enough of LATTICE to exercise build_obj_knn without loading a dataset."""

    def __init__(self, feats, tb=None):
        self.n_items = feats.shape[0]
        grp = torch.unique(feats, dim=0, return_inverse=True)[1]
        self.obj_grp = grp
        self.obj_first = torch.tensor(
            [int((grp == g).nonzero()[0]) for g in torch.unique(grp)])
        self.obj_counts = torch.bincount(grp)
        self._tb = tb
        self._eps_used = None

    build_obj_knn = Models.LATTICE.build_obj_knn
    _knn_reserve = Models.LATTICE._knn_reserve
    _knn_group = Models.LATTICE._knn_group
    _knn_threshold = Models.LATTICE._knn_threshold
    _tiebreak_eps = Models.LATTICE._tiebreak_eps


def _rand_tb(n, seed=0):
    """The same symmetric random matrix `--knn_tiebreak random` builds.

    Symmetry matters: an asymmetric tiebreak (a per-column ramp, say) orders every row the same
    way, so a whole tie group still picks the same neighbours and the hub survives.
    """
    r = torch.rand(n, n, generator=torch.Generator().manual_seed(seed))
    return 0.5 * (r + r.t())          # scaled into [0, 1], as _build_tiebreak does


def tie_feats():
    """8 items in 4 tie groups of sizes 4/2/1/1, in a geometry with distinct group similarities."""
    base = torch.tensor([[1.0, 0.0, 0.0],
                         [0.9, 0.4, 0.0],
                         [0.2, 1.0, 0.0],
                         [0.0, 0.1, 1.0]])
    return base[[0, 0, 0, 0, 1, 1, 2, 3]].clone()


# ---------------------------------------------------------------------------------------------

def test_default_bit_identical():
    print("\ndefault path is bit-identical")
    torch.manual_seed(0)
    for name, adj in [("random", torch.randn(40, 40)),
                      ("with exact ties", build_sim(tie_feats()))]:
        st = Stub(build_sim(tie_feats()) if name != "random" else torch.randn(40, 3))
        with Cfg(obj_knn='default', knn_tiebreak='none', knn_self='keep',
                 knn_selfloop_alpha=0.0, knn_sym='none', topk=3):
            a = build_knn_neighbourhood(adj, 3)
            b = st.build_obj_knn.__func__(st, adj) if False else None
        # build_obj_knn takes features, so compare the op it delegates to instead.
        c = _topk_from_scores(adj, adj, 3)
        check(f"topk_from_scores == build_knn_neighbourhood ({name})", torch.equal(a, c))

    f = tie_feats().requires_grad_(True)
    with Cfg(obj_knn='default', knn_tiebreak='none', knn_self='keep',
             knn_selfloop_alpha=0.0, knn_sym='none', topk=3):
        out = Stub(tie_feats()).build_obj_knn.__func__(Stub(tie_feats()), f)
    ref = build_knn_neighbourhood(build_sim(f), 3)
    check("build_obj_knn(default) == published construction", torch.equal(out, ref))
    g1 = torch.autograd.grad(out.sum(), f, retain_graph=True)[0]
    g2 = torch.autograd.grad(ref.sum(), f)[0]
    check("gradients match", torch.equal(g1, g2))


def test_tiebreak():
    print("\ntiebreak")
    f = tie_feats()
    n = f.shape[0]
    st = Stub(f, tb=_rand_tb(n))
    with Cfg(obj_knn='tiebreak', knn_tiebreak='random', knn_tiebreak_eps=0.0,
             knn_self='keep', knn_selfloop_alpha=0.0, knn_sym='none', topk=3):
        out = st.build_obj_knn.__func__(st, f)
    adj = build_sim(f)
    nz = out != 0
    check("exactly topk edges per row", bool((nz.sum(1) == 3).all()))
    check("no _NEG sentinel leaked into a value", float(out.min()) > -1.0,
          f"min={float(out.min()):.4f}")
    check("values come from the unperturbed similarity",
          torch.equal(out[nz], adj[nz]))
    # Group 0 has 4 members and topk=3, so with a safe eps every slot must stay inside it.
    grp = st.obj_grp
    same = grp.unsqueeze(1) == grp.unsqueeze(0)
    rows = (grp == grp[0])
    check("a group larger than topk keeps every slot in-group",
          bool((nz[rows] & ~same[rows]).sum() == 0))
    eps = st._eps_used
    counts = torch.bincount(st.obj_grp)
    gap = topk_boundary_gap(build_sim(f[st.obj_first]), counts, 3)
    check("eps below the provable safety bound", eps <= 0.499 * gap + 1e-12,
          f"eps={eps:.4g} bound={0.499 * gap:.4g}")
    # The guarantee is on the multiset of similarity *values*, not on which item fills a slot.
    base_vals = torch.sort(build_knn_neighbourhood(adj, 3), dim=-1, descending=True)[0][:, :3]
    tb_vals = torch.sort(out, dim=-1, descending=True)[0][:, :3]
    check("the top-k similarity values are unchanged, only their owners",
          torch.allclose(base_vals, tb_vals, atol=1e-6),
          f"max|diff|={float((base_vals - tb_vals).abs().max()):.3g}")
    with Cfg(obj_knn='tiebreak', knn_tiebreak='random', knn_tiebreak_eps=0.0,
             knn_self='keep', knn_selfloop_alpha=0.0, knn_sym='none', topk=3):
        again = st.build_obj_knn.__func__(st, f)
    check("deterministic across repeat calls", torch.equal(out, again))

    # De-hubbing is the whole point, and it only has room to show when a tie group is much
    # larger than topk -- which is the real situation (largest group 711, topk 10). With 24
    # members and topk=3 the default sends all 24 rows to the same 3 columns.
    big = tie_feats()[[0] * 24 + [1, 2, 3]]
    st2 = Stub(big, tb=_rand_tb(big.shape[0]))
    common = dict(knn_self='keep', knn_selfloop_alpha=0.0, knn_sym='none', topk=3)
    with Cfg(obj_knn='default', knn_tiebreak='none', **common):
        base = st2.build_obj_knn.__func__(st2, big)
    with Cfg(obj_knn='tiebreak', knn_tiebreak='random', knn_tiebreak_eps=0.0, **common):
        tb_out = st2.build_obj_knn.__func__(st2, big)
    b_in, t_in = (base != 0).sum(0), (tb_out != 0).sum(0)
    # Mean in-degree is topk by construction; a healthy graph keeps the max within a small
    # multiple of it instead of concentrating every row on the same few columns.
    check("de-hubs a large tie group",
          int(t_in.max()) <= 3 * common['topk'] and int(t_in.max()) < int(b_in.max()) / 2,
          f"max in-deg {int(t_in.max())} vs {int(b_in.max())}, mean {common['topk']}")
    check("removes the zero-in-degree tail",
          float((t_in == 0).float().mean()) < float((b_in == 0).float().mean()),
          f"zero-in {float((t_in == 0).float().mean()):.2f} vs "
          f"{float((b_in == 0).float().mean()):.2f}")


def test_self_and_selfloop():
    print("\nself-loop controls")
    f = tie_feats()
    n = f.shape[0]
    st = Stub(f, tb=_rand_tb(n))
    with Cfg(obj_knn='tiebreak', knn_tiebreak='random', knn_tiebreak_eps=0.0,
             knn_self='drop', knn_selfloop_alpha=0.0, knn_sym='none', topk=3):
        out = st.build_obj_knn.__func__(st, f)
    check("--knn_self drop removes every self-loop", float(out.diagonal().abs().max()) == 0.0)
    check("still topk edges per row", bool(((out != 0).sum(1) == 3).all()))
    with Cfg(obj_knn='default', knn_tiebreak='none', knn_self='keep',
             knn_selfloop_alpha=0.25, knn_sym='none', topk=3):
        out = st.build_obj_knn.__func__(st, f)
    base = build_knn_neighbourhood(build_sim(f), 3)
    check("--knn_selfloop_alpha adds only alpha*I",
          torch.allclose(out - base, 0.25 * torch.eye(n)))


def test_reserve():
    print("\nreserve")
    f = tie_feats()
    n = f.shape[0]
    st = Stub(f, tb=_rand_tb(n))
    grp = st.obj_grp
    same = grp.unsqueeze(1) == grp.unsqueeze(0)
    with Cfg(obj_knn='reserve', knn_tiebreak='random', knn_tiebreak_eps=0.0, knn_reserve=1,
             knn_self='keep', knn_selfloop_alpha=0.0, knn_sym='none', topk=3):
        out = st.build_obj_knn.__func__(st, f)
    nz = out != 0
    in_grp = (nz & same).sum(1)
    want = torch.tensor([min(1, int((grp == g).sum())) for g in grp])
    check("exactly min(r, |group|) same-group slots", bool((in_grp == want).all()),
          f"{in_grp.tolist()} vs {want.tolist()}")
    check("exactly topk slots total", bool((nz.sum(1) == 3).all()), str(nz.sum(1).tolist()))
    check("no _NEG leaked", float(out.min()) > -1.0)
    with Cfg(obj_knn='reserve', knn_tiebreak='random', knn_tiebreak_eps=0.0, knn_reserve=9,
             knn_self='keep', knn_selfloop_alpha=0.0, knn_sym='none', topk=3):
        out = st.build_obj_knn.__func__(st, f)
    check("r > topk is clamped, not an error", bool(((out != 0).sum(1) <= 3).all()))


def test_group():
    print("\ngroup")
    f = tie_feats()
    n = f.shape[0]
    st = Stub(f, tb=_rand_tb(n))
    grp = st.obj_grp
    with Cfg(obj_knn='group', knn_tiebreak='random', knn_tiebreak_eps=0.0, knn_group_m=1,
             knn_self='keep', knn_selfloop_alpha=0.0, knn_sym='none', topk=3):
        out = st.build_obj_knn.__func__(st, f)
    nz = out != 0
    g = build_sim(f[st.obj_first])
    top1 = torch.topk(g - 2 * torch.eye(g.shape[0]), 1, dim=-1)[1].squeeze(1)
    for i in range(n):
        allowed = {int(grp[i]), int(top1[grp[i]])}
        bad = [int(j) for j in nz[i].nonzero().flatten() if int(grp[j]) not in allowed]
        if bad:
            check(f"row {i} restricted to own + top-m groups", False, f"leaked to {bad}")
            return
    check("every row restricted to own group + top-m groups", True)
    check("no _NEG leaked", float(out.min()) > -1.0)


def test_threshold():
    print("\nthreshold")
    f = tie_feats()
    n = f.shape[0]
    st = Stub(f)
    adj = build_sim(f)
    t = 0.8
    with Cfg(obj_knn='threshold', knn_threshold=t, knn_tiebreak='none', knn_self='keep',
             knn_selfloop_alpha=0.0, knn_sym='none', topk=3):
        out = st.build_obj_knn.__func__(st, f)
    keep = adj >= t
    check("an edge is kept iff cos >= threshold", torch.equal(out != 0, keep),
          f"{int((out != 0).sum())} edges vs {int(keep.sum())} expected")
    check("kept values are the unperturbed similarity", torch.equal(out[keep], adj[keep]))
    check("out-degree is not pinned to topk", bool(((out != 0).sum(1) != 3).any()),
          str((out != 0).sum(1).tolist()))
    check("density is recorded for the log",
          abs(st._obj_density - float(keep.float().mean())) < 1e-6)

    # --knn_self drop writes _NEG on the diagonal; the threshold must read that as "no edge"
    # rather than comparing the sentinel against 0.8 and keeping adj's real 1.0.
    with Cfg(obj_knn='threshold', knn_threshold=t, knn_tiebreak='none', knn_self='drop',
             knn_selfloop_alpha=0.0, knn_sym='none', topk=3):
        out_ns = st.build_obj_knn.__func__(st, f)
    check("--knn_self drop composes with threshold", float(out_ns.diagonal().abs().max()) == 0.0)
    off = ~torch.eye(n, dtype=torch.bool)
    check("dropping self changes nothing off the diagonal", torch.equal(out_ns[off], out[off]))

    # A threshold above the largest off-diagonal similarity isolates rows. The Laplacian must
    # leave them as exact zeros: rowsum 0 -> pow(-0.5) = inf -> the guard maps it to 0.
    with Cfg(obj_knn='threshold', knn_threshold=1.0, knn_tiebreak='none', knn_self='drop',
             knn_selfloop_alpha=0.0, knn_sym='none', topk=3):
        empty = st.build_obj_knn.__func__(st, f)
    with Cfg(fast_laplacian=0):
        lap = compute_normalized_laplacian(empty)
    iso = (empty != 0).sum(1) == 0
    check("threshold can isolate rows", bool(iso.any()), f"{int(iso.sum())}/{n} isolated")
    check("isolated rows stay exactly zero through the Laplacian",
          bool((lap[iso] == 0).all()) and bool(torch.isfinite(lap).all()))


def test_sym():
    print("\nsymmetrise")
    torch.manual_seed(1)
    a = torch.rand(20, 20)
    a = build_knn_neighbourhood(a, 3)
    m = symmetrise(a, 'max')
    check("max is symmetric", torch.equal(m, m.t()))
    check("max only adds edges", bool(((a != 0) <= (m != 0)).all()))
    check("max leaves no zero in-degree where out-degree > 0",
          bool(((m != 0).sum(0) > 0).all()))
    check("none is a no-op", torch.equal(symmetrise(a, 'none'), a))


def test_laplacian():
    print("\nfast Laplacian")
    torch.manual_seed(2)
    for name, a in [("dense random", torch.rand(200, 200)),
                    ("sparsified", build_knn_neighbourhood(torch.rand(200, 200), 10)),
                    ("a zero row", torch.cat([torch.zeros(1, 200),
                                              torch.rand(199, 200)], 0))]:
        with Cfg(fast_laplacian=0):
            slow = compute_normalized_laplacian(a.clone())
        with Cfg(fast_laplacian=1):
            fast = compute_normalized_laplacian(a.clone())
        check(f"bit-identical ({name})", torch.equal(slow, fast),
              f"max|diff|={float((slow - fast).abs().max()):.3g}")


def test_real_tensors():
    print("\nfast Laplacian on the real cached tensors")
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    for p in sorted((ROOT / "data" / "home_v2-2" / "5-core").glob("*_adj_10.pt")):
        a = torch.load(p, map_location=dev)
        with Cfg(fast_laplacian=0):
            slow = compute_normalized_laplacian(a)
        with Cfg(fast_laplacian=1):
            fast = compute_normalized_laplacian(a)
        check(f"bit-identical ({p.name})", torch.equal(slow, fast),
              f"max|diff|={float((slow - fast).abs().max()):.3g}")
        del a, slow, fast
        if dev == 'cuda':
            torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real", action="store_true",
                   help="also verify against data/home_v2-2/5-core/*_adj_10.pt (read-only)")
    a = p.parse_args(_REAL_ARGV[1:])

    test_default_bit_identical()
    test_tiebreak()
    test_self_and_selfloop()
    test_reserve()
    test_group()
    test_threshold()
    test_sym()
    test_laplacian()
    if a.real:
        test_real_tensors()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failures: " + ", ".join(FAIL))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
