import os
import numpy as np
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from utility.parser import parse_args
from utility.norm import build_sim, build_knn_normalized_graph
args = parse_args()

class MICRO(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, weight_size, dropout_list, image_feats, text_feats, graph_feats=None):
        super().__init__()
        self.use_obj = 'graph' in [m.strip() for m in args.modalities.split(',')]
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
            

        image_adj = build_sim(self.image_embedding.weight.detach())
        image_adj = build_knn_normalized_graph(image_adj, topk=args.topk, is_sparse=args.sparse, norm_type=args.norm_type)

        text_adj = build_sim(self.text_embedding.weight.detach())
        text_adj = build_knn_normalized_graph(text_adj, topk=args.topk, is_sparse=args.sparse, norm_type=args.norm_type)

        self.text_original_adj = text_adj.cuda()
        self.image_original_adj = image_adj.cuda()
        
        self.image_trs = nn.Linear(image_feats.shape[1], args.embed_size)
        self.text_trs = nn.Linear(text_feats.shape[1], args.embed_size)

        self.softmax = nn.Softmax(dim=-1)


        self.query = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )

        self.tau = 0.5

        # Built last, and only when the object modality is on, so that the two-modality default
        # leaves every other module's initialisation drawing from an unshifted RNG stream.
        if self.use_obj:
            self.graph_embedding = nn.Embedding.from_pretrained(torch.Tensor(graph_feats), freeze=False)
            graph_adj = build_sim(self.graph_embedding.weight.detach())
            graph_adj = build_knn_normalized_graph(graph_adj, topk=args.topk, is_sparse=args.sparse, norm_type=args.norm_type)
            self.graph_original_adj = graph_adj.cuda()
            self.graph_trs = nn.Linear(graph_feats.shape[1], args.embed_size)

    def mm(self, x, y):
        if args.sparse:
            return torch.sparse.mm(x, y)
        else:
            return torch.mm(x, y)
    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def batched_contrastive_loss(self, z1, z2, batch_size=4096):
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))
                   
        loss_vec = torch.cat(losses)
        return loss_vec.mean()

    def forward(self, adj, build_item_graph=False):
        image_feats = self.image_trs(self.image_embedding.weight)
        text_feats = self.text_trs(self.text_embedding.weight)
        if build_item_graph:
            self.image_adj = build_sim(image_feats) 
            self.image_adj = build_knn_normalized_graph(self.image_adj, topk=args.topk, is_sparse=args.sparse, norm_type=args.norm_type)
            self.image_adj = (1 - args.lambda_coeff) * self.image_adj + args.lambda_coeff * self.image_original_adj

            self.text_adj = build_sim(text_feats) 
            self.text_adj = build_knn_normalized_graph(self.text_adj, topk=args.topk, is_sparse=args.sparse, norm_type=args.norm_type)
            self.text_adj = (1 - args.lambda_coeff) * self.text_adj + args.lambda_coeff * self.text_original_adj

        else:
            self.image_adj = self.image_adj.detach()
            self.text_adj = self.text_adj.detach()

        image_item_embeds = self.item_id_embedding.weight
        text_item_embeds = self.item_id_embedding.weight

        for i in range(args.layers):
            image_item_embeds = self.mm(self.image_adj, image_item_embeds)

        for i in range(args.layers):
            text_item_embeds = self.mm(self.text_adj, text_item_embeds)

        # fusion_embeds feed the attention; contrast_embeds are what main.py runs InfoNCE over.
        # They differ only for the --graph_mode diagnostic arms.
        fusion_embeds = [image_item_embeds, text_item_embeds]
        contrast_embeds = [image_item_embeds, text_item_embeds]

        if self.use_obj:
            graph_feats = self.graph_trs(self.graph_embedding.weight)
            if build_item_graph:
                self.graph_adj = build_sim(graph_feats)
                self.graph_adj = build_knn_normalized_graph(self.graph_adj, topk=args.topk, is_sparse=args.sparse, norm_type=args.norm_type)
                self.graph_adj = (1 - args.lambda_coeff) * self.graph_adj + args.lambda_coeff * self.graph_original_adj
            else:
                self.graph_adj = self.graph_adj.detach()

            graph_item_embeds = self.item_id_embedding.weight
            for i in range(args.layers):
                graph_item_embeds = self.mm(self.graph_adj, graph_item_embeds)

            if args.graph_mode in ('both', 'fusion'):
                fusion_embeds.append(graph_item_embeds)
            if args.graph_mode in ('both', 'contrast'):
                contrast_embeds.append(graph_item_embeds)

        att = torch.cat([self.query(z) for z in fusion_embeds], dim=-1)
        weight = self.softmax(att)
        h = sum(weight[:, k].unsqueeze(dim=1) * z for k, z in enumerate(fusion_embeds))

        
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
            i_g_embeddings = i_g_embeddings + F.normalize(h, p=2, dim=1)
            return u_g_embeddings, i_g_embeddings, contrast_embeds, h

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
            i_g_embeddings = i_g_embeddings + F.normalize(h, p=2, dim=1)
            return u_g_embeddings, i_g_embeddings, contrast_embeds, h

        elif args.cf_model == 'mf':
                return self.user_embedding.weight, self.item_id_embedding.weight + F.normalize(h, p=2, dim=1), contrast_embeds, h


class MF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, weight_size, dropout_list, image_feats=None, text_feats=None):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, adj, build_item_graph=False):
        return self.user_embedding.weight, self.item_embedding.weight



class NGCF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, weight_size, dropout_list, image_feats=None, text_feats=None):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.weight_size = weight_size
        self.n_ui_layers = len(self.weight_size)
        self.dropout_list = nn.ModuleList()
        self.GC_Linear_list = nn.ModuleList()
        self.Bi_Linear_list = nn.ModuleList()

        self.weight_size = [self.embedding_dim] + self.weight_size
        for i in range(self.n_ui_layers):
            self.GC_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i+1]))
            self.Bi_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i+1]))
            self.dropout_list.append(nn.Dropout(dropout_list[i]))

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_id_embedding = nn.Embedding(n_items, embedding_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

    def forward(self, adj, build_item_graph):
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

        all_embeddings = torch.cat(all_embeddings, dim=1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings

class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, weight_size, dropout_list, image_feats=None, text_feats=None):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_ui_layers = len(weight_size)

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_id_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

    def forward(self, adj, build_item_graph):
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings