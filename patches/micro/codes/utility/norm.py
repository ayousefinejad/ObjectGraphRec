import torch

def build_sim(context):
    context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
    sim = torch.mm(context_norm, context_norm.transpose(1, 0))
    return sim

def build_knn_normalized_graph(adj, topk, is_sparse, norm_type):
    device = adj.device
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    if is_sparse:
        # Vectorised rewrite of the original construction, which was
        #   tuple_list = [[row, int(col)] for row in range(len(knn_ind)) for col in knn_ind[row]]
        # -- a Python loop over n_items * topk entries with an int(col) device->host sync on each
        # one (145,030 of them here, and this is called twice per epoch). The edge list it built
        # is exactly (arange(N) repeated topk times, knn_ind.flatten()) in that order, so this is
        # the same COO tensor in the same order, without leaving the GPU.
        n = knn_ind.shape[0]
        row = torch.arange(n, device=device).repeat_interleave(topk)
        col = knn_ind.flatten()
        i = torch.stack([row, col], dim=0)
        v = knn_val.flatten()
        edge_index, edge_weight = get_sparse_laplacian(i, v, normalization=norm_type, num_nodes=adj.shape[0])
        return torch.sparse_coo_tensor(edge_index, edge_weight, adj.shape)
    else:
        weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
        return get_dense_laplacian(weighted_adjacency_matrix, normalization=norm_type)

def get_sparse_laplacian(edge_index, edge_weight, num_nodes, normalization='none'):
    row, col = edge_index[0], edge_index[1]
    # Was torch_scatter.scatter_add(edge_weight, row, dim=0, dim_size=num_nodes), which is not
    # installed here and needs a compiled extension matched to the torch build. index_add_ into a
    # pre-sized zero vector is the same reduction; it is also differentiable, which matters because
    # edge_weight carries a gradient on the build_item_graph=True pass.
    deg = torch.zeros(num_nodes, dtype=edge_weight.dtype, device=edge_weight.device)
    deg = deg.index_add(0, row, edge_weight)

    if normalization == 'sym':
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    elif normalization == 'rw':
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = deg_inv[row] * edge_weight
    return edge_index, edge_weight


def get_dense_laplacian(adj, normalization='none'):
    if normalization == 'sym':
        rowsum = torch.sum(adj, -1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
        L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    elif normalization == 'rw':
        rowsum = torch.sum(adj, -1)
        d_inv = torch.pow(rowsum, -1)
        d_inv[torch.isinf(d_inv)] = 0.
        d_mat_inv = torch.diagflat(d_inv)
        L_norm = torch.mm(d_mat_inv, adj)
    elif normalization == 'none':
        L_norm = adj
    return L_norm
