import json
import os
import numpy as np
from time import time

import torch
import torch.nn as nn
import torch.sparse as sparse
import torch.nn.functional as F

from utility.parser import parse_args
args = parse_args()

def build_knn_neighbourhood(adj, topk):
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
    return weighted_adjacency_matrix
def compute_normalized_laplacian(adj):
    rowsum = torch.sum(adj, -1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    if args.fast_laplacian:
        # D^-1/2 A D^-1/2 with D diagonal is just a row and a column rescale. The published form
        # materialises the n x n diagonal and does two dense 14503^3 matmuls (22.7 s); this is
        # 0.2 s and *bit-identical*, because each entry of D@A is one nonzero product plus a sum
        # of exact zeros. Verified with torch.equal on the real cached tensors, not allclose.
        return adj * d_inv_sqrt.unsqueeze(1) * d_inv_sqrt.unsqueeze(0)
    d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
    L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return L_norm
def build_sim(context):
    context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
    sim = torch.mm(context_norm, context_norm.transpose(1, 0))
    return sim


# --------------------------------------------------------------------------------------------
# Item-graph propagation with learned attention (--item_prop gat).
#
# Hand-written rather than torch_geometric.nn.GATConv for the same reason ObjectGraph/encoder.py
# hand-writes WeightedSAGEConv: the library module drops the edge weight. Here that is not a
# nuisance but a correctness requirement --- `h = item_adj @ h` is the ONLY gradient path from
# the loss back to modal_weight / image_trs / text_trs / graph_trs / gate. A conv that consumes
# only edge_index (integer, non-differentiable) turns the entire modality fusion into a silent
# no-op that still trains and still produces plausible numbers. This project already lost time
# to exactly that shape of bug once, in GraphMAE stage 2.
#
# So the fused edge weight multiplies the message:
#
#     out_i = sum_j  alpha_ij * w_ij * (W x_j)
#
# with w_ij gathered from item_adj *with autograd live*. d out / d w is then a first-order term,
# not one routed through a learned edge encoder that could decay to zero mid-run.
# --------------------------------------------------------------------------------------------

def _segment_softmax(logits, dst, n):
    """softmax of `logits` within each dst group. Groups with no members stay all-zero.

    Max-subtracted for numerical stability. `n` is passed explicitly because shape inference
    from `dst.max()` would silently drop trailing nodes that have no in-edges.
    """
    shape = (n,) + logits.shape[1:]
    m = torch.full(shape, float('-inf'), device=logits.device, dtype=logits.dtype)
    m = m.scatter_reduce(0, dst.view(-1, *([1] * (logits.dim() - 1))).expand_as(logits),
                         logits, reduce='amax', include_self=True)
    m = torch.nan_to_num(m, neginf=0.0)                       # empty groups -> 0, exp(0-0)=1
    e = torch.exp(logits - m[dst])
    den = torch.zeros(shape, device=logits.device, dtype=logits.dtype).index_add_(0, dst, e)
    return e / den[dst].clamp_min(1e-16)


class ItemGATConv(nn.Module):
    """Multi-head attention over the fused item graph, weighted by the fused edge value.

    At ``init='reduce'`` (W = I, a = 0) attention is uniform over each neighbourhood, so the
    layer computes ``(1/deg_i) * sum_j w_ij x_j`` --- exactly the published ``item_adj @ h`` up
    to the per-row constant that ``F.normalize`` divides out downstream. The published operator
    is therefore inside this module's hypothesis class, which is what makes an exact reduction
    test possible; the arm starts at the published model and learns away from it.
    """

    def __init__(self, dim, heads=4, dropout=0.0, act='none', residual=0, init='reduce'):
        super().__init__()
        if dim % heads:
            raise ValueError('embed dim %d not divisible by heads %d' % (dim, heads))
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        self.dropout, self.act, self.residual = dropout, act, bool(residual)
        self.lin = nn.Linear(dim, dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.empty(heads, self.head_dim))
        if init == 'reduce':
            with torch.no_grad():
                self.lin.weight.copy_(torch.eye(dim))          # concat of heads rebuilds I
            nn.init.zeros_(self.a_src)
            nn.init.zeros_(self.a_dst)                          # -> uniform attention
        else:
            nn.init.xavier_uniform_(self.lin.weight)
            nn.init.xavier_uniform_(self.a_src)
            nn.init.xavier_uniform_(self.a_dst)
        self._alpha_stats = {}

    def forward(self, x, src, dst, w, n):
        xh = self.lin(x).view(n, self.heads, self.head_dim)
        logits = F.leaky_relu((xh[src] * self.a_src).sum(-1) + (xh[dst] * self.a_dst).sum(-1),
                              negative_slope=0.2)
        alpha = _segment_softmax(logits, dst, n)                # [E, heads]
        with torch.no_grad():
            # Reported so a flat downstream result can be told apart from attention that never
            # moved. Per *node*, averaged over nodes that have in-edges, so it is directly
            # comparable to log(mean in-degree) -- the value uniform attention would give. A
            # sum over edges would move with graph size and mean nothing on its own.
            p = alpha.clamp_min(1e-12)
            ent = torch.zeros(n, device=x.device, dtype=x.dtype).index_add_(
                0, dst, -(p * p.log()).mean(-1))
            deg = torch.zeros(n, device=x.device, dtype=x.dtype).index_add_(
                0, dst, torch.ones_like(w))
            has = deg > 0
            self._alpha_stats = {
                'gat_alpha_entropy': float(ent[has].mean()),
                'gat_uniform_entropy': float(deg[has].log().mean()),   # the no-op reference
                'gat_max_alpha': float(alpha.max())}
        if self.dropout and self.training:
            alpha = F.dropout(alpha, p=self.dropout, training=True)
        msg = xh[src] * (alpha * w.unsqueeze(1)).unsqueeze(-1)  # w keeps the fusion gradient
        out = torch.zeros(n, self.heads, self.head_dim, device=x.device, dtype=x.dtype)
        out = out.index_add_(0, dst, msg).reshape(n, self.dim)
        if self.residual:
            out = out + x
        return F.elu(out) if self.act == 'elu' else out


# --------------------------------------------------------------------------------------------
# Tie-aware top-k for the object modality.
#
# `build_knn_neighbourhood` above is untouched and stays the default path. The variants below
# exist because the object features are not merely coarse but degenerate: 531 unique rows over
# 14,503 items. Inside a tie group every candidate similarity is *exactly* equal, so torch.topk
# falls back to lowest index and all 711 members of the largest group choose the same 10
# neighbours. The encoder's inter-label geometry is never reached because the ties saturate
# top-10 before a different label appears.
# --------------------------------------------------------------------------------------------

_NEG = -1e9


def _topk_from_scores(adj, scores, topk):
    """Rank by `scores`, but keep the *unperturbed* `adj` values as edge weights.

    Splitting rank from value is what makes the tiebreak safe: the graph the model sees carries
    real object similarities, the perturbation only decides which of several exactly-equal
    candidates gets the slot. `scores` is always a constant w.r.t. autograd, so no gradient path
    is created from the tiebreak signal into the object graph.
    """
    knn_sc, knn_ind = torch.topk(scores, topk, dim=-1)
    knn_val = torch.gather(adj, -1, knn_ind)
    # A row with fewer allowed candidates than topk still returns topk indices, the surplus ones
    # carrying the _NEG sentinel. Those must become no edge, not an edge with adj's real value.
    knn_val = torch.where(knn_sc > _NEG / 2, knn_val, torch.zeros_like(knn_val))
    return torch.zeros_like(adj).scatter_(-1, knn_ind, knn_val)


def topk_boundary_gap(g, counts, topk):
    """How much room a tiebreak has before it changes *which similarities* enter the top-k.

    Rows inside a tie group are identical, so the full n x n similarity matrix takes its values
    from the n_groups x n_groups matrix `g`, each entry repeated `counts` times. For one row,
    sort those repeated values, read off the k-th largest v_k, and measure down to the next
    strictly smaller distinct value. A perturbation smaller than that gap can permute members of
    the v_k plateau -- which is the entire point -- but cannot pull in a similarity that the
    published top-k excluded. The multiset of *values* in each neighbourhood is preserved
    exactly; only the choice among equal ones changes.

    The naive alternative -- the smallest positive gap anywhere in `g` -- is useless here: two of
    the 531 groups sit ~1e-8 apart, which would force eps below float32 resolution at cos ~ 1
    and make the tiebreak a no-op.
    """
    order = torch.argsort(g, dim=-1, descending=True)
    vals = torch.gather(g, -1, order)
    cum = counts[order].cumsum(-1)
    idx = (cum >= topk).float().argmax(-1, keepdim=True)        # block holding rank topk-1
    v_k = vals.gather(-1, idx)
    below = torch.where(vals < v_k, vals, torch.full_like(vals, -2.0)).max(-1, keepdim=True)[0]
    return float((v_k - below).min())


def symmetrise(adj, mode):
    if mode == 'max':
        return torch.maximum(adj, adj.t())
    if mode == 'mean':
        return 0.5 * (adj + adj.t())
    return adj


def knn_diagnostics(adj, grp):
    """Structure of a sparsified item-item graph, from its nonzero pattern alone.

    Reads the same off a raw kNN adjacency or off a cached normalized Laplacian: D^-1/2 A D^-1/2
    is a row/column rescale, so it preserves the nonzero pattern wherever the degree is nonzero.
    Every arm prints one of these; an arm whose numbers equal the control's did not take effect.
    """
    nz = adj != 0
    n = nz.shape[0]
    in_deg = nz.sum(0).float()
    out_deg = nz.sum(1)
    slots = float(out_deg.sum())
    # Row-signature hash, chunked: the n x n float64 product would be 1.7 GB.
    h = torch.randn(n, dtype=torch.float64, device=nz.device, generator=_gen(nz.device, 12345))
    sig = torch.cat([(nz[i:i + 1024].double() * h).sum(1) for i in range(0, n, 1024)])
    d = {
        'n_items': n,
        'zero_in_frac': round(float((in_deg == 0).float().mean()), 4),
        'max_in_deg': int(in_deg.max()),
        'p99_in_deg': int(torch.quantile(in_deg, 0.99)),
        'distinct_nbhd': int(torch.unique(sig).numel()),
        'mutual_frac': round(float((nz & nz.t()).sum()) / max(slots, 1), 4),
        'selfloop_rows': round(float(nz.diagonal().float().mean()), 4),
        'mean_out_deg': round(slots / n, 3),
    }
    if grp is not None:
        same = grp.unsqueeze(1) == grp.unsqueeze(0)
        d['dup_slot_frac'] = round(float((nz & same).sum()) / max(slots, 1), 4)
        d['n_tie_groups'] = int(torch.unique(grp).numel())
    return d


def _gen(device, seed):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


def _obj_spec():
    """Non-default object-kNN settings only, so the published run yields an empty dict.

    Emptiness is the cache key and the bit-identity contract in one: no entries means every
    knob is at its published value, which means the cached adjacency is still valid.
    """
    s = {}
    if args.obj_knn != 'default':
        s['mode'] = args.obj_knn
    if args.obj_knn == 'threshold':
        s['threshold'] = args.knn_threshold
    if args.knn_tiebreak != 'none':
        s['tiebreak'] = args.knn_tiebreak
        if args.knn_tiebreak_eps:
            s['eps'] = args.knn_tiebreak_eps
        if args.knn_tiebreak == 'random':
            s['tb_seed'] = args.knn_tb_seed
    if args.knn_reserve:
        s['reserve'] = args.knn_reserve
    if args.obj_knn == 'group':
        s['group_m'] = args.knn_group_m
    if args.knn_self != 'keep':
        s['self'] = args.knn_self
    if args.knn_selfloop_alpha:
        s['selfloop_alpha'] = args.knn_selfloop_alpha
    return s


_FROZEN = os.path.realpath('data/home_v2-2')


def _under_frozen(path):
    """True if `path` resolves inside the shipped tree, which is never written."""
    r = os.path.realpath(path)
    return r == _FROZEN or r.startswith(_FROZEN + os.sep)


def load_or_build_adj(modality, spec, build_fn):
    """Cache the modality adjacency, but only when the construction is the published one.

    The cached `*_adj_10.pt` files were built with the default top-k rule. Any --obj_knn/--knn_*
    flag changes that construction, so a cache hit would silently score the published graph
    instead of the arm's -- and with lambda_coeff=0.9 that is 90% of the item graph. A rebuild
    costs ~1.5 s against a ~1400 s run once --fast_laplacian is on, so non-default specs simply
    never read and never write a cache: a cache that does not exist cannot go stale.
    """
    path = 'data/%s/%s-core/%s_adj_%d.pt' % (args.dataset, args.core, modality, args.topk)
    default = not spec
    if os.path.exists(path):
        if not default:
            raise RuntimeError(
                '%s exists but %s is built with a non-default spec %s. Loading it would score '
                'the published graph under the arm\'s name. Point --dataset at a directory that '
                'does not link this cache.' % (path, modality, json.dumps(spec, sort_keys=True)))
        adj = torch.load(path)
        source = 'cache'
    else:
        adj = build_fn()
        if default:
            if _under_frozen(path):
                raise RuntimeError('refusing to write %s: inside the frozen data/home_v2-2 tree'
                                   % path)
            torch.save(adj, path)
            source = 'built'
        else:
            source = 'rebuilt'
    print('LATTICE_KNN %s source=%s path=%s spec=%s'
          % (modality, source, os.path.realpath(path), json.dumps(spec, sort_keys=True)),
          flush=True)
    return adj

class LATTICE(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, weight_size, dropout_list, image_feats, text_feats, graph_feats):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.weight_size = weight_size
        self.n_ui_layers = len(self.weight_size)
        self.weight_size = [self.embedding_dim] + self.weight_size
        self.user_embedding = nn.Embedding(n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        if args.cf_model == 'ngcf':
            self.GC_Linear_list = nn.ModuleList()
            self.Bi_Linear_list = nn.ModuleList()
            self.dropout_list = nn.ModuleList()
            for i in range(self.n_ui_layers):
                self.GC_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i+1]))
                self.Bi_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i+1]))
                self.dropout_list.append(nn.Dropout(dropout_list[i]))


        self.image_embedding = nn.Embedding.from_pretrained(torch.Tensor(image_feats), freeze=False)
        self.text_embedding = nn.Embedding.from_pretrained(torch.Tensor(text_feats), freeze=False)
        self.graph_embedding = nn.Embedding.from_pretrained(torch.Tensor(graph_feats), freeze=False)
            
    
        # Tie groups of the *raw* object features. A linear projection preserves exact row
        # equality, so this partition is also the tie structure of the learned branch -- it is
        # computed once and reused for the tiebreak epsilon, for `reserve`/`group`, and for the
        # duplicate-slot diagnostic.
        uniq, inv, counts = np.unique(graph_feats, axis=0, return_inverse=True, return_counts=True)
        self.register_buffer('obj_grp', torch.as_tensor(inv, dtype=torch.long).view(-1))
        self.register_buffer('obj_first', torch.as_tensor(
            np.unique(inv, return_index=True)[1], dtype=torch.long))
        self.register_buffer('obj_counts', torch.as_tensor(counts, dtype=torch.long))
        self._obj_spec = _obj_spec()
        self._sym_spec = {} if args.knn_sym == 'none' else {'sym': args.knn_sym}
        self._tb = self._build_tiebreak(text_feats)
        self._eps_used = None

        def _plain(emb):
            return lambda: compute_normalized_laplacian(
                symmetrise(build_knn_neighbourhood(build_sim(emb.weight.detach()), args.topk),
                           args.knn_sym))

        image_adj = load_or_build_adj('image', self._sym_spec, _plain(self.image_embedding))
        text_adj = load_or_build_adj('text', self._sym_spec, _plain(self.text_embedding))
        graph_adj = load_or_build_adj(
            'graph', {**self._sym_spec, **self._obj_spec},
            lambda: compute_normalized_laplacian(
                self.build_obj_knn(self.graph_embedding.weight.detach())))

        self.text_original_adj = text_adj.cuda()
        self.image_original_adj = image_adj.cuda()
        self.graph_original_adj = graph_adj.cuda()
        print('LATTICE_DIAG ' + json.dumps(
            {'arm_spec': {**self._sym_spec, **self._obj_spec}, 'eps_used': self._eps_used,
             **knn_diagnostics(self.graph_original_adj, self.obj_grp.cuda())}, sort_keys=True),
            flush=True)

        self.modality_mask = torch.Tensor(
            [float(m in args.modalities.split(',')) for m in ('image', 'text', 'graph')]).cuda()

        self.image_trs = nn.Linear(image_feats.shape[1], args.feat_embed_dim)
        self.text_trs = nn.Linear(text_feats.shape[1], args.feat_embed_dim)
        self.graph_trs = nn.Linear(graph_feats.shape[1], args.feat_embed_dim)


        self.modal_weight = nn.Parameter(torch.Tensor([1/3, 1/3, 1/3]))
        self.softmax = nn.Softmax(dim=0)

        # Modality fusion. 'softmax' is the published behaviour and the default: three global
        # scalars summing to one, applied to the item-item adjacencies. The alternatives relax
        # that in two steps -- 'sigmoid' drops the zero-sum constraint, 'gated'/'conf' make the
        # weights per-item. The object modality is what motivates per-item: its embeddings are
        # heavily tied (293 unique vectors across 14,503 items, largest group 956), so how much
        # signal it carries varies item to item and one global scalar cannot express that.
        if args.fusion in ('gated', 'conf'):
            self.gate = nn.Linear(3 * args.feat_embed_dim, 3)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)          # sigmoid(0)=0.5 -> starts as a flat gate
        if args.fusion == 'conf':
            # Per-item reliability of the object channel, free from the features themselves: an
            # item whose object vector is shared with 955 others carries almost no information.
            conf = 1.0 / np.log1p(counts[inv])
            self.register_buffer('obj_conf',
                                 torch.Tensor(conf / conf.max()).view(-1, 1))

        # Constructed LAST, and only in gat mode. nn.Linear and xavier_uniform_ draw from the
        # global RNG, so building these earlier would shift the init stream of every parameter
        # after them and silently break comparability with the arms already measured. In dense
        # mode nothing is allocated, no RNG is drawn, and no state-dict key is added.
        self.item_gat = None
        if args.item_prop == 'gat':
            # RNG state is saved and restored around construction. nn.Linear draws from the
            # global stream in its constructor (even under init='reduce', where the draw is
            # immediately overwritten by the identity), which would shift every later consumer
            # -- including the negative sampler -- and give this arm a different data order than
            # the control. Restoring makes the arm differ from the control in the propagation
            # operator and nothing else.
            _rng = torch.get_rng_state()
            _rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            self.item_gat = nn.ModuleList([
                ItemGATConv(args.embed_size, heads=args.gat_heads, dropout=args.gat_dropout,
                            act=args.gat_act, residual=args.gat_residual, init=args.gat_init)
                for _ in range(args.n_layers)])
            torch.set_rng_state(_rng)
            if _rng_cuda is not None:
                torch.cuda.set_rng_state_all(_rng_cuda)
        # Ablate the *weighting*, not the modality encoders: modal_weight (and the per-item gate)
        # stop learning, while image_trs / text_trs / graph_trs keep their gradients. Frozen at
        # the uniform initialisation, so this arm is "every modality counted equally, forever".
        self._frozen_fusion = []
        if args.freeze_fusion:
            self.modal_weight.requires_grad_(False)
            self._frozen_fusion.append('modal_weight')
            if hasattr(self, 'gate'):
                for p in self.gate.parameters():
                    p.requires_grad_(False)
                self._frozen_fusion.append('gate')
        # Its own provenance line, for the reason --item_prop needed one: a silently ignored flag
        # would leave this arm numerically identical to the control and nothing would say so.
        print('LATTICE_FUSE mode=%s freeze=%d frozen=[%s]' % (
            args.fusion, int(bool(args.freeze_fusion)), ','.join(self._frozen_fusion)), flush=True)
        print('LATTICE_PROP mode=%s layers=%d%s' % (
            args.item_prop, args.n_layers,
            '' if args.item_prop != 'gat' else
            ' heads=%d head_dim=%d act=%s residual=%d dropout=%s init=%s' % (
                args.gat_heads, args.embed_size // args.gat_heads, args.gat_act,
                args.gat_residual, args.gat_dropout, args.gat_init)), flush=True)

    # ----------------------------------------------------------------------------------------
    # Object-modality kNN
    # ----------------------------------------------------------------------------------------

    def _build_tiebreak(self, text_feats):
        """The n x n constant used to order items *inside* a tie group, or None.

        Raw text cosine, deliberately not the projected features: the projection heads are
        near-random (they see a gradient on 1 batch in 150), and a differentiable tiebreak would
        open a gradient path from text_trs into the object graph that the published model does
        not have. `random` is the control -- it de-hubs just as well with zero cross-modal
        information, so it separates "the tiebreak carries text signal" from "the tiebreak
        breaks the hubs".

        Scaled into [0, 1] either way, so that a weight of eps moves any score by at most eps and
        the safety bound in `_tiebreak_eps` is exact.
        """
        if args.knn_tiebreak == 'none':
            return None
        n = self.n_items
        if args.knn_tiebreak == 'text':
            return 0.5 * (build_sim(torch.Tensor(text_feats)) + 1.0)
        r = torch.rand(n, n, generator=_gen('cpu', args.knn_tb_seed))
        return 0.5 * (r + r.t())                            # symmetric, so i<->j agree on order

    def _tiebreak_eps(self, feats):
        """Largest tiebreak weight that cannot change which similarities enter the top-k.

        `self._tb` is scaled into [0, 1], so a weight of eps moves any score by at most eps and
        the bound is exactly the top-k boundary gap.
        """
        gap = topk_boundary_gap(build_sim(feats[self.obj_first]), self.obj_counts, args.topk)
        safe = 0.499 * gap
        eps = args.knn_tiebreak_eps or safe
        if eps > safe:
            print('LATTICE_WARN tiebreak eps %.3g exceeds the safe bound %.3g: it can pull in '
                  'similarities the published top-k excluded, not only permute tied ones'
                  % (eps, safe), flush=True)
        if eps < 1e-9:
            # build_obj_knn offsets each row by its max before adding eps*tb, so the usable
            # resolution is ~1e-45 rather than the 1.2e-7 of float32 at cos=1. This threshold is
            # therefore about a degenerate similarity matrix, not about arithmetic.
            print('LATTICE_WARN tiebreak eps %.3g leaves no room to order tied candidates; two '
                  'object groups must be almost exactly as similar as a group is to itself'
                  % eps, flush=True)
        self._eps_used = float(eps)          # exact, not rounded: it is a reproducibility record
        return eps

    def build_obj_knn(self, feats):
        """Sparsify the object similarity matrix under the arm's top-k rule.

        Used for both branches -- the original adjacency built from the raw features and the
        learned one built from graph_trs output -- because a linear projection preserves exact
        row equality, so the learned branch inherits the same tie groups and the same hubs.
        Repairing only one of them would leave 90% of the item graph (lambda_coeff=0.9) broken.
        """
        adj = build_sim(feats)
        if args.obj_knn == 'default' and args.knn_selfloop_alpha == 0 and args.knn_self == 'keep':
            out = build_knn_neighbourhood(adj, args.topk)
            return symmetrise(out, args.knn_sym)

        scores = adj
        if args.knn_tiebreak != 'none':
            if self._tb.device != adj.device:
                self._tb = self._tb.to(adj.device)      # once, not per rebuild
            # Subtract the row max first. The safe eps is ~1e-6 while float32 resolves only
            # 1.2e-7 near cos=1, so adding it to the raw similarity would quantise the tiebreak
            # away -- exactly the failure this change exists to avoid. Offsetting by a per-row
            # constant leaves the within-row ordering untouched (IEEE subtraction is monotone)
            # and puts the top plateau at 0, where float32 resolves ~1e-45.
            scores = ((adj - adj.max(-1, keepdim=True).values)
                      + self._tiebreak_eps(feats) * self._tb)
        if args.knn_self == 'drop':
            scores = scores.clone()
            scores.fill_diagonal_(_NEG)

        if args.obj_knn == 'reserve':
            out = self._knn_reserve(adj, scores)
        elif args.obj_knn == 'group':
            out = self._knn_group(adj, scores, feats)
        elif args.obj_knn == 'threshold':
            out = self._knn_threshold(adj, scores)
        else:
            out = _topk_from_scores(adj, scores, args.topk)

        if args.knn_selfloop_alpha:
            out = out + args.knn_selfloop_alpha * torch.eye(
                out.shape[0], device=out.device, dtype=out.dtype)
        # The learned branch rebuilds this graph from graph_trs output once an epoch, and a fixed
        # absolute cosine cut on a projection that is still moving can go near-empty or
        # near-complete without changing any metric visibly. LATTICE_DIAG covers the original
        # branch, which is fixed; this covers the other one. Cheap: one pass over a matrix that
        # was just materialised anyway.
        self._obj_density = round(float((out != 0).float().mean()), 6)
        return symmetrise(out, args.knn_sym)

    def _knn_threshold(self, adj, scores):
        """Every pair at or above --knn_threshold, no k at all.

        The other modes all keep exactly topk slots and only argue about which ones; this one
        drops that constraint, so out-degree varies with how many similar items an item actually
        has (mean 269 on the tuned encoder, median 128, max 1767) and 0.6% of items end up
        isolated. Isolated rows are safe: compute_normalized_laplacian zeroes the inf from
        rowsum^-0.5, so the row stays exactly zero rather than becoming NaN.

        Ranking on `scores` and taking values from `adj` matches _topk_from_scores. The _NEG
        sentinel that --knn_self drop writes on the diagonal is far below any admissible
        threshold, so the two compose without a special case.
        """
        return torch.where(scores >= args.knn_threshold, adj, torch.zeros_like(adj))

    def _knn_reserve(self, adj, scores):
        """r slots from the item's own tie group, topk-r from other groups.

        Forces the encoder's inter-label geometry into the neighbourhood even when the tie group
        is large enough to fill top-k on its own.
        """
        r = min(args.knn_reserve, args.topk)
        same = self.obj_grp.unsqueeze(1) == self.obj_grp.unsqueeze(0)
        near = _topk_from_scores(adj, torch.where(same, scores, scores.new_full((), _NEG)), r)
        far = _topk_from_scores(adj, torch.where(same, scores.new_full((), _NEG), scores),
                                args.topk - r)
        # The two supports are disjoint by construction (same group vs different group), so the
        # sum cannot double-count a slot. A group with fewer than r members simply contributes
        # fewer than r edges -- _topk_from_scores drops the sentinel slots.
        return near + far

    def _knn_group(self, adj, scores, feats):
        """Candidates restricted to the item's own tie group plus the top-m *groups*.

        kNN over the 531 unique vectors, expanded back to items -- the coarsest way to make the
        neighbourhood depend on label-level similarity rather than on item index.
        """
        g = build_sim(feats[self.obj_first])
        m = min(args.knn_group_m, g.shape[0] - 1)
        keep = torch.zeros_like(g, dtype=torch.bool)
        if m:
            keep.scatter_(-1, torch.topk(g.fill_diagonal_(_NEG), m, dim=-1)[1], True)
        keep.fill_diagonal_(True)
        allowed = keep[self.obj_grp][:, self.obj_grp]
        return _topk_from_scores(adj, torch.where(allowed, scores, scores.new_full((), _NEG)),
                                 args.topk)

    def _modal_weights(self, image_feats, text_feats, graph_feats):
        """Either 3 global scalars (softmax/sigmoid) or an n_items x 3 per-item gate."""
        if args.fusion == 'softmax':
            # Masking the logits rather than the weights keeps the survivors summing to one, so
            # --modalities image,text gives exactly [0.5, 0.5, 0] instead of a rescaled graph.
            w = self.modal_weight.masked_fill(self.modality_mask == 0, float('-inf'))
            return self.softmax(w)
        if args.fusion == 'sigmoid':
            return torch.sigmoid(self.modal_weight) * self.modality_mask
        g = torch.sigmoid(self.gate(torch.cat([image_feats, text_feats, graph_feats], dim=1)))
        if args.fusion == 'conf':
            g = torch.cat([g[:, :2], g[:, 2:] * self.obj_conf], dim=1)
        return g * self.modality_mask

    def _rho_diag(self, cf, gh):
        """Record what rho actually buys, in norm terms.

        rho is not a mixing ratio: ||h/||h|||| is exactly 1 by construction while ||x_bar||
        is unconstrained and grows through training, so the effective contribution of the graph
        path is rho / ||x_bar||. Reporting rho alone would hide that the same rho means something
        different at epoch 0 and epoch 120.
        """
        with torch.no_grad():
            cf_n = float(cf.norm(dim=1).mean())
            self._rho_stats = {'rho': args.rho, 'cf_norm': round(cf_n, 4),
                               'graph_norm': round(float(gh.norm(dim=1).mean()), 4),
                               'rho_effective': round(args.rho / cf_n, 4) if cf_n else None}

    def fusion_state(self):
        """What the fusion parameters actually learned, for the log. No effect on the model.

        Without this, a flat `--lr_fusion` arm has two indistinguishable readings: the weights
        moved and uniform turned out to be near-optimal (a result), or the weights never moved
        and the arm measured nothing (a bug). `logit_range` separates them -- it is 0.0 at init
        by construction, since modal_weight starts at [1/3, 1/3, 1/3].

        The `*_trs_norm` entries ask the same question of the parts of the fusion that are not a
        single 3-vector. They are absolute norms rather than drifts, which is enough because the
        line is printed at every eval including epoch 0 -- the epoch-0 row *is* the baseline.
        """
        with torch.no_grad():
            mw = self.modal_weight.detach()
            out = {'modal_weight': [round(float(v), 4) for v in mw],
                   'softmax': [round(float(v), 4) for v in self.softmax(mw)],
                   'logit_range': round(float(mw.max() - mw.min()), 4)}
            for name in ('image_trs', 'text_trs', 'graph_trs'):
                w = getattr(self, name).weight
                out[name + '_norm'] = round(float(w.norm()), 4)
            out.update(getattr(self, '_rho_stats', {}))
            if hasattr(self, 'gate'):
                # Initialised to exactly zero, so its norm *is* its drift.
                out['gate_norm'] = round(float(self.gate.weight.norm()), 4)
            if self.item_gat is not None:
                # Same argument as gate_norm and logit_range: a flat itemgat arm has two
                # readings -- attention learned something and it did not help (a result), or
                # attention never moved (a bug). a_src/a_dst start at exactly zero and W at
                # exactly I, so these norms *are* the drift, and alpha_entropy leaves log(deg)
                # only if the coefficients stopped being uniform.
                a = torch.cat([l.a_src.reshape(-1) for l in self.item_gat] +
                              [l.a_dst.reshape(-1) for l in self.item_gat])
                eye = torch.eye(args.embed_size, device=a.device)
                out['gat_a_norm'] = round(float(a.norm()), 4)
                out['gat_w_drift'] = round(float(sum((l.lin.weight - eye).norm()
                                                     for l in self.item_gat)), 4)
                out.update({k: round(v, 4)
                            for k, v in (self.item_gat[-1]._alpha_stats or {}).items()})
            if args.obj_knn == 'threshold':
                # Density of the *learned* object graph, whose out-degree is not pinned to topk
                # and is free to drift with graph_trs. Constant under every other mode, so it is
                # only worth a column here.
                out['obj_density'] = getattr(self, '_obj_density', None)
            return out

    def _fuse(self, mats, w):
        """Weighted sum of the per-modality adjacencies.

        A per-item gate is applied as sqrt(g_i g_j) * A_ij rather than row-scaling, so the result
        stays symmetric and compute_normalized_laplacian below is still well defined. Note that
        that Laplacian is exactly invariant to a global rescaling of A, so only per-item gates
        (not a uniform sigmoid) change the learned branch; the original branch is not
        re-normalised after mixing, so magnitude matters there.
        """
        out = None
        for k, m in enumerate(mats):
            if w.dim() == 1:
                term = m * w[k]
            else:
                s = torch.sqrt(w[:, k]).unsqueeze(1)
                term = (m * s) * s.t()
            out = term if out is None else out + term
        return out

    def forward(self, adj, build_item_graph=False):
        image_feats = self.image_trs(self.image_embedding.weight)
        text_feats = self.text_trs(self.text_embedding.weight)
        graph_feats = self.graph_trs(self.graph_embedding.weight)
        if build_item_graph:
            # Drop last epoch's adjacencies BEFORE allocating this epoch's. Every one of these
            # four is overwritten a few lines below, but `self.x = f(...)` holds the old tensor
            # alive until the new one exists, so the peak carried 4 x 841 MB of dead weight into
            # the rebuild -- which is what OOMed at 804 MiB when a second tenant on this GPU grew
            # to 9.8 GB. Releasing a reference to a tensor that is about to be replaced reads no
            # value and changes no arithmetic; the bit-identity check is in the commit message.
            self.item_adj = None
            self.image_adj = self.text_adj = self.graph_adj = None
            weight = self._modal_weights(image_feats, text_feats, graph_feats)
            self.image_adj = build_sim(image_feats)
            self.image_adj = build_knn_neighbourhood(self.image_adj, topk=args.topk)

            self.image_adj = symmetrise(self.image_adj, args.knn_sym)

            self.text_adj = build_sim(text_feats)
            self.text_adj = build_knn_neighbourhood(self.text_adj, topk=args.topk)
            self.text_adj = symmetrise(self.text_adj, args.knn_sym)

            # The learned branch gets the same top-k rule as the original one: graph_trs is
            # linear, so it maps identical rows to identical rows and the projected object
            # features carry exactly the same 531 tie groups.
            self.graph_adj = self.build_obj_knn(graph_feats)


            learned_adj = self._fuse([self.image_adj, self.text_adj, self.graph_adj], weight)
            learned_adj = compute_normalized_laplacian(learned_adj)
            original_adj = self._fuse([self.image_original_adj, self.text_original_adj,
                                       self.graph_original_adj], weight)
            self.item_adj = (1 - args.lambda_coeff) * learned_adj + args.lambda_coeff * original_adj
        else:
            self.item_adj = self.item_adj.detach()

        if self.item_gat is None:
            h = self.item_id_embedding.weight
            for i in range(args.n_layers):
                h = torch.mm(self.item_adj, h)
        else:
            # Edges are cached and detached on exactly the same schedule as item_adj above, so
            # the fusion parameters keep the gradient schedule they have today -- live on batch
            # 0 of each epoch, detached on the rest -- and no more.
            if build_item_graph:
                with torch.no_grad():
                    # row = dst, col = src. torch_geometric's dense_to_sparse returns the
                    # transpose of this and would propagate item_adj.t(); item_adj is asymmetric
                    # whenever knn_sym='none', which is the default.
                    dst, src = self.item_adj.nonzero(as_tuple=True)
                self._gat_src, self._gat_dst = src, dst
                self._gat_w = self.item_adj[dst, src]      # differentiable: the fusion gradient
                if torch.is_grad_enabled() and self.item_adj.requires_grad:
                    assert self._gat_w.requires_grad, (
                        'item-graph edge weights carry no gradient: every fusion parameter '
                        '(modal_weight, *_trs, gate) would be a silent no-op')
            else:
                self._gat_w = self._gat_w.detach()
            h = self.item_id_embedding.weight
            for layer in self.item_gat:
                h = layer(h, self._gat_src, self._gat_dst, self._gat_w, self.n_items)

        # With cf_model='mf' the modalities reach a score *only* through item_adj topology, i.e.
        # through a rank-10 quantization of the content space that is rebuilt once an epoch.
        # This gives them a dense path that gets a gradient on every batch. Off by default: it is
        # a departure from LATTICE, not a reimplementation of it.
        extra = 0.
        if args.content_path_beta:
            w = self._modal_weights(image_feats, text_feats, graph_feats)
            content = sum(m * (w[k] if w.dim() == 1 else w[:, k:k + 1])
                          for k, m in enumerate([image_feats, text_feats, graph_feats]))
            extra = args.content_path_beta * F.normalize(content, p=2, dim=1)

        if args.cf_model == 'ngcf':
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
            all_embeddings = [ego_embeddings]
            for i in range(self.n_ui_layers):
                side_embeddings = torch.sparse.mm(adj, ego_embeddings)
                sum_embeddings = F.leaky_relu(self.GC_Linear_list[i](side_embeddings))
                bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
                bi_embeddings = F.leaky_relu(self.Bi_Linear_list[i](bi_embeddings))
                ego_embeddings = sum_embeddings + bi_embeddings
                ego_embeddings = self.dropout_list[i](ego_embeddings)

                norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
                all_embeddings += [norm_embeddings]

            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = all_embeddings.mean(dim=1, keepdim=False)            
            u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
            i_g_embeddings = i_g_embeddings + args.rho * F.normalize(h, p=2, dim=1) + extra
            return u_g_embeddings, i_g_embeddings
        elif args.cf_model == 'lightgcn':
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
            all_embeddings = [ego_embeddings]
            for i in range(self.n_ui_layers):
                side_embeddings = torch.sparse.mm(adj, ego_embeddings)
                ego_embeddings = side_embeddings
                all_embeddings += [ego_embeddings]
            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
            u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
            i_g_embeddings = i_g_embeddings + args.rho * F.normalize(h, p=2, dim=1) + extra
            return u_g_embeddings, i_g_embeddings
        elif args.cf_model == 'mf':
                gh = args.rho * F.normalize(h, p=2, dim=1)
                self._rho_diag(self.item_id_embedding.weight, gh)
                return self.user_embedding.weight, self.item_id_embedding.weight + gh + extra

