# coding: utf-8
# @email: daiji23323@gmail.com
r"""
crane
# Update: 15/12/2024
"""

import json
import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss
from utils.utils import build_sim, compute_normalized_laplacian


class crane(GeneralRecommender):
    def __init__(self, config, dataset):
        super(crane, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.knn_k = config['knn_k']
        self.lambda_coeff = config['lambda_coeff']
        self.cf_model = config['cf_model']
        self.n_layers = config['n_mm_layers']
        self.n_ui_layers = config['n_ui_layers']
        self.reg_weight = config['reg_weight']
        self.build_item_graph = True
        self.mm_image_weight = config['mm_image_weight']
        self.dropout = config['dropout']
        self.degree_ratio = config['degree_ratio']

        self.n_nodes = self.n_users + self.n_items

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.masked_adj, self.mm_adj = None, None
        self.edge_indices, self.edge_values = self.get_edge_info()
        self.edge_indices, self.edge_values = self.edge_indices.to(self.device), self.edge_values.to(self.device)
        self.edge_full_indices = torch.arange(self.edge_values.size(0)).to(self.device)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        mm_adj_file = os.path.join(dataset_path, 'mm_adj_cranedsp_{}_{}.pt'.format(self.knn_k, int(10*self.mm_image_weight)))

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file)
        else:
            if self.v_feat is not None:
                indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
                self.mm_adj = image_adj
            if self.t_feat is not None:
                indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                self.mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
                del text_adj
                del image_adj
            torch.save(self.mm_adj, mm_adj_file)

        # Object graph. Off by default, so CRANE.yaml alone still reproduces the published model.
        self.object_graph = None
        self.o_feat = None
        if config['use_object_graph']:
            o_path = os.path.join(dataset_path, config['object_feature_file'])
            if not os.path.isfile(o_path):
                raise FileNotFoundError(
                    'use_object_graph is on but %s is missing. Export it with '
                    'object-graph/scripts/export_mmrec.py.' % o_path)
            self.o_feat = torch.from_numpy(
                np.load(o_path, allow_pickle=True)).type(torch.FloatTensor).to(self.device)
            # Built with the same knn_k and the same top-k + symmetric-Laplacian rule as the
            # image and text graphs above, via the model's own get_knn_adj_mat -- LATTICE also
            # gives all three modalities one shared args.topk rather than a per-modality k.
            # Symmetrising is off by default, so the published/original behaviour is unchanged.
            # It is in the cache filename because a directed and a symmetric graph are different
            # tensors: sharing one filename would silently serve whichever was built first.
            # config[...] returns None for an absent key (configurator.py:117-121), so this is
            # False both when the key is missing and when it is set to false. Config is not a
            # dict and has no .get().
            self.obj_knn_sym = bool(config['obj_knn_sym'])
            obj_adj_file = os.path.join(dataset_path, 'obj_adj_{}{}.pt'.format(
                self.knn_k, '_sym' if self.obj_knn_sym else ''))
            self.object_graph = ObjectGraph(
                self.o_feat, self.feat_embed_dim, n_layers=config['n_obj_layers'],
                residual=bool(config['obj_residual']))
            if os.path.exists(obj_adj_file):
                obj_adj = torch.load(obj_adj_file).to(self.device)
            else:
                if self.obj_knn_sym:
                    # top-k is directed: each item picks 10 neighbours, but because 99% of items
                    # sit in a tie group and topk breaks ties by index, the in-edges pile onto the
                    # lowest-indexed member of each group. Measured on this graph: 78.5% of items
                    # have in-degree 0, i.e. the object channel sends them nothing at all, and
                    # with residual=True they pass through untouched. Adding the reverse edge
                    # makes every item a receiver (reach 21.5% -> 100%) and lowers the worst hub
                    # from 711 to 721 in-edges spread over the whole catalogue rather than 21% of
                    # it. This is LATTICE's --knn_sym, which CRANE never had.
                    idx, _ = self.get_knn_adj_mat(
                        self.object_graph.object_embedding.weight.detach())
                    idx = torch.cat([idx, idx.flip(0)], dim=1)
                    n = self.object_graph.object_embedding.weight.shape[0]
                    obj_adj = self.compute_normalized_laplacian(idx, (n, n))
                else:
                    _, obj_adj = self.get_knn_adj_mat(
                        self.object_graph.object_embedding.weight.detach())
                torch.save(obj_adj, obj_adj_file)
            self.object_graph.set_adj(obj_adj)

        # Cross-modal attention network
        # cross_modal_batch_first defaults to False = the published behaviour, bit-for-bit.
        # See CrossModalAttention for what the two settings actually compute -- and note that a
        # third modality is only meaningful on the batch_first=True axis, where the attention
        # runs over modalities. On the published axis it would only widen the batch dimension
        # from 2 to 3 and the object features would never meet the image or text ones.
        self.object_in_attention = bool(
            self.object_graph is not None and config['object_in_attention'])
        if self.object_in_attention and not config['cross_modal_batch_first']:
            raise ValueError(
                'object_in_attention requires cross_modal_batch_first: on the published axis the '
                'modalities are the batch dimension, so a third one is never attended over.')
        self.cross_modal_attention = CrossModalAttention(
            self.feat_embed_dim, self.embedding_dim,
            batch_first=bool(config['cross_modal_batch_first']),
            n_modalities=3 if self.object_in_attention else 2)

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def get_norm_adj_mat(self):
        # Upstream built a dok_matrix and filled it via the private A._update(data_dict), which
        # scipy has since removed (1.16.3 here). dok is still a dict subclass but its storage
        # moved, so dict.update(A, ...) silently no-ops and would give an all-zero adjacency
        # rather than an error. Built directly as COO instead: the two blocks are disjoint
        # (upper-right and lower-left) so no coordinate repeats, which makes COO's
        # duplicate-summing identical to the dict's overwrite. Checked against the dok reference.
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        n = self.n_users + self.n_items
        row = np.concatenate([inter_M.row, inter_M_t.row + self.n_users])
        col = np.concatenate([inter_M.col + self.n_users, inter_M_t.col])
        A = sp.coo_matrix((np.ones(len(row), dtype=np.float32), (row, col)),
                          shape=(n, n)).tocsr()
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def pre_epoch_processing(self):
        if self.dropout <= .0:
            self.masked_adj = self.norm_adj
            return
        # degree-sensitive edge pruning
        degree_len = int(self.edge_values.size(0) * (1. - self.dropout))
        degree_idx = torch.multinomial(self.edge_values, degree_len)
        # random sample
        keep_indices = self.edge_indices[:, degree_idx]
        # norm values
        keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.n_users, self.n_items)))
        all_values = torch.cat((keep_values, keep_values))
        # update keep_indices to users/items+self.n_users
        keep_indices[1] += self.n_users
        all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
        self.masked_adj = torch.sparse.FloatTensor(all_indices, all_values, self.norm_adj.shape).to(self.device)

    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        # edge normalized values
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))
        return edges, values

    def forward(self, adj):
        # Cross-modal attention fusion
        if self.v_feat is not None and self.t_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
            text_feats = self.text_trs(self.text_embedding.weight)
            object_feats = self.object_graph.features() if self.object_in_attention else None
            fused_feats = self.cross_modal_attention(image_feats, text_feats, object_feats)
        else:
            fused_feats = self.item_id_embedding.weight

        # Object graph, in front of the latent feature learning module below: it shapes the item
        # representations that the mm_adj propagation then consumes, rather than being merged
        # into mm_adj itself. Identity when use_object_graph is off.
        if self.object_graph is not None:
            fused_feats = self.object_graph(fused_feats)

        # Item-item graph convolution
        h = fused_feats
        for i in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)

        # User-item graph convolution
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        return u_g_embeddings, i_g_embeddings + h

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        return mf_loss

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings = self.forward(self.masked_adj)
        self.build_item_graph = False

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
        return batch_mf_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores


class CrossModalAttention(nn.Module):
    # combined_feats below is (n_items, 2, d), and nn.MultiheadAttention reads its input as
    # (seq, batch, embed) unless batch_first=True. So the two settings compute different things:
    #
    #   batch_first=False (published)  seq=n_items, batch=2 -> each modality attends across all
    #                                  14,503 ITEMS. Not cross-modal at all; the two modalities
    #                                  never see each other inside the attention.
    #   batch_first=True  (intended)   seq=2, batch=n_items -> each item attends across its own
    #                                  two MODALITIES, which is what the paper describes.
    #
    # Either way the output is (n_items, 2, d) and the .mean(dim=1) below reduces it to
    # (n_items, d), so the shape contract downstream is unchanged and the bug is silent.
    # Default False keeps the published behaviour bit-for-bit; True is offered as a diagnostic.
    def __init__(self, feat_embed_dim, embedding_dim, batch_first=False, n_modalities=2):
        super(CrossModalAttention, self).__init__()
        self.feat_embed_dim = feat_embed_dim
        self.embedding_dim = embedding_dim

        # Projection layers for visual and textual features
        self.visual_proj = nn.Linear(feat_embed_dim, embedding_dim)
        self.textual_proj = nn.Linear(feat_embed_dim, embedding_dim)
        # Constructed only when the object modality is on. nn.Linear draws from the global RNG,
        # so creating it unconditionally would shift every subsequent init and the two-modality
        # model would stop being bit-identical to the published one at a fixed seed.
        self.object_proj = nn.Linear(feat_embed_dim, embedding_dim) if n_modalities == 3 else None

        # Attention layers
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=4,
                                               batch_first=batch_first)

    def forward(self, visual_feats, textual_feats, object_feats=None):
        # Project features to the same dimension
        visual_proj = self.visual_proj(visual_feats)
        textual_proj = self.textual_proj(textual_feats)

        # Concatenate features for cross-modal attention. torch.stack(dim=1) is exactly the
        # cat-of-unsqueeze this replaced, so the two-modality output is unchanged.
        parts = [visual_proj, textual_proj]
        if self.object_proj is not None:
            parts.append(self.object_proj(object_feats))
        combined_feats = torch.stack(parts, dim=1)

        # Apply multi-head attention
        # need_weights=False is a pure memory optimisation, not a semantic change: the weights are
        # discarded on the next line anyway, and setting it False lets PyTorch use the fused
        # scaled_dot_product_attention path instead of materialising the attention matrix. That
        # matters on the published batch_first=False path, where this input is read as
        # seq_len=n_items, batch=2 -- the matrix is (2*4, 14503, 14503) = 6.7 GB, and
        # another 6.7 GB for its gradient. On the authors' Baby split (~7k items) it fits; at
        # 14,503 items it does not. Output values are identical either way.
        attn_output, _ = self.attention(combined_feats, combined_feats, combined_feats,
                                        need_weights=False)
        fused_feats = attn_output.mean(dim=1)

        return fused_feats


class ObjectGraph(nn.Module):
    """LATTICE's object modality, as a stage in front of CRANE's latent feature learning.

    LATTICE (Models.py in the object-graph repo) treats the object features as a third modality
    alongside image and text: a frozen embedding table -> a linear projection (`graph_trs`) -> a
    top-k cosine kNN item-item graph -> a symmetric normalised Laplacian, fused with the image
    and text adjacencies and then propagated over. This is the same construction, kept in
    CRANE's sparse idiom (CRANE stores its kNN graphs as sparse top-k tensors rather than
    LATTICE's dense 14503^2 / 841 MB matrices) and placed *before* the mm_adj propagation
    instead of merged into it, so the two graphs stay separable and the object stage can be
    ablated on its own.

    Two things are deliberately not copied from LATTICE:

    * **The learned branch.** LATTICE rebuilds the kNN graph from the *projected* features once
      an epoch and blends it with the frozen one as `(1-lambda)*learned + lambda*original`.
      CRANE dropped that -- it builds mm_adj once at init and never touches it, which is why
      `lambda_coeff` and `build_item_graph` are assigned in crane.__init__ and then never read.
      This module follows CRANE, not LATTICE, so the object graph is frozen too and the image,
      text and object graphs are all built the same way. Restoring the learned branch would be a
      change to all three, not to this one.
    * **The tiebreak machinery.** See the degeneracy note in `set_adj`.

    The projected features are also offered to the cross-modal attention via `features()`, which
    is the other half of "a third modality": the object channel reaches the score both through
    the graph topology here and through the fused content vector.
    """

    def __init__(self, o_feat, feat_embed_dim, n_layers=1, residual=True):
        super(ObjectGraph, self).__init__()
        self.object_embedding = nn.Embedding.from_pretrained(o_feat, freeze=False)
        self.object_trs = nn.Linear(o_feat.shape[1], feat_embed_dim)
        self.n_layers = n_layers
        self.residual = residual
        self.adj = None

    def set_adj(self, adj):
        self.adj = adj
        # The object features are heavily tied -- on home_v2's default_fixed encoder, 531 unique
        # rows across 14,503 items with the largest group at 711, i.e. 99.2% of items share their
        # vector with at least one other. Every member of a group has cosine 1.0 with every other,
        # so torch.topk falls back to its index order and all 711 select the *same* 10 neighbours:
        # the graph collapses onto the lowest-indexed member of each group. That is the published
        # LATTICE behaviour (its `--obj_knn default` arm) and is kept here so this is comparable
        # to the LATTICE obj_lgn3 runs, but it is a property of the graph, not a bug in it, and it
        # bounds what the object channel can contribute. Logged rather than silently accepted.
        idx = adj.coalesce().indices()
        in_deg = torch.bincount(idx[1], minlength=self.object_embedding.weight.shape[0])
        print('CRANE_OBJGRAPH ' + json.dumps({
            'nnz': int(idx.shape[1]),
            'zero_in_degree_frac': round(float((in_deg == 0).float().mean()), 4),
            'max_in_degree': int(in_deg.max()),
            'n_layers': self.n_layers, 'residual': self.residual,
        }, sort_keys=True), flush=True)

    def features(self):
        """Projected object features, for the cross-modal attention's third token."""
        return self.object_trs(self.object_embedding.weight)

    def forward(self, h):
        # n_layers=0 is the ablation that keeps the attention token but removes the graph, so it
        # has to be the exact identity. Without this the residual below would return h + h = 2h,
        # rescaling the content branch against the id embedding it is later added to and making
        # the ablation a second change rather than one.
        if self.n_layers == 0:
            return h
        out = h
        for _ in range(self.n_layers):
            out = torch.sparse.mm(self.adj, out)
        # Additive by default. LATTICE never replaces an embedding with its propagated version
        # either -- it returns `i_g_embeddings + F.normalize(h)`. It matters more here than there:
        # with 99.2% of items in a tie group, a pure replacement would overwrite each item's fused
        # content vector with an average over its tie group and discard what the attention built.
        return h + out if self.residual else out

# utils/utils.py:40 resolves the class as getattr(models.<name.lower()>, <name>), while
# utils/configurator.py:76 resolves the config as configs/model/<name>.yaml. The repo ships
# CRANE.yaml but names the class `crane`, so neither `-m CRANE` (no such attribute) nor
# `-m crane` (no such yaml -- every hyperparameter would silently read None) works as documented.
# This alias makes the README's `-m CRANE` resolve both. No model code is changed.
CRANE = crane
