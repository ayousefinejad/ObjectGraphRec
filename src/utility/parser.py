import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--data_path', nargs='?', default='../data/',
                        help='Input data path.')
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed')
    parser.add_argument('--dataset', nargs='?', default='baby',
                        help='Choose a dataset from {sports, baby, clothing}')
    parser.add_argument('--verbose', type=int, default=5,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=200,
                        help='Number of epoch.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]',
                        help='Regularizations.')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate.')
    parser.add_argument('--model_name', nargs='?', default='lattice',
                        help='Specify the model name.')

    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--feat_embed_dim', type=int, default=64,
                        help='')                        
    parser.add_argument('--weight_size', nargs='?', default='[64,64]',
                        help='Output sizes of every layer')
    parser.add_argument('--core', type=int, default=5,
                        help='5-core for warm-start; 0-core for cold start')
    parser.add_argument('--topk', type=int, default=10,
                        help='K value of k-NN sparsification')  
    parser.add_argument('--lambda_coeff', type=float, default=0.9,
                        help='Lambda value of skip connection')
    parser.add_argument('--cf_model', nargs='?', default='lightgcn',
                        help='Downstream Collaborative Filtering model {mf, ngcf, lightgcn}')
    parser.add_argument('--fusion', nargs='?', default='softmax',
                        help='Modality fusion {softmax, sigmoid, gated, conf}. softmax is the '
                             'published behaviour: three global scalars summing to one.')
    # --- object-graph kNN construction ------------------------------------------------------
    # The object modality's features are heavily tied (531 unique vectors across 14,503 items),
    # and torch.topk breaks exact ties by lowest index, so every member of a tie group ends up
    # aggregating from the same few lowest-indexed items: 78.2% of items have in-degree 0 and only
    # 531 distinct neighbourhoods exist. These flags select alternative top-k rules that spread
    # the choice inside a tie group. All of them default to the published behaviour.
    parser.add_argument('--obj_knn', nargs='?', default='default',
                        choices=['default', 'tiebreak', 'reserve', 'group', 'threshold'],
                        help='Object-modality kNN rule. default is the published torch.topk.')
    parser.add_argument('--knn_threshold', type=float, default=0.0,
                        help='--obj_knn threshold: keep every pair with cosine >= this value and '
                             'ignore --topk entirely, so an item gets as many neighbours as it has '
                             'similar ones. On the tuned encoder 0.8 gives a mean out-degree of '
                             '269 against the top-k rule 10, and leaves 0.6% of items isolated.')
    parser.add_argument('--knn_tiebreak', nargs='?', default='none',
                        choices=['none', 'text', 'random'],
                        help='Signal used to order items inside an exact tie group.')
    parser.add_argument('--knn_tiebreak_eps', type=float, default=0.0,
                        help='Tiebreak weight. 0 = auto, set to 0.499x the smallest positive gap '
                             'between two distinct object similarities, which provably cannot '
                             'reorder distinct groups.')
    parser.add_argument('--knn_tb_seed', type=int, default=0,
                        help='Seed for --knn_tiebreak random.')
    parser.add_argument('--knn_reserve', type=int, default=0,
                        help='--obj_knn reserve: slots reserved for the item own tie group.')
    parser.add_argument('--knn_group_m', type=int, default=0,
                        help='--obj_knn group: candidates restricted to own group + top-m groups.')
    parser.add_argument('--knn_self', nargs='?', default='keep', choices=['keep', 'drop'],
                        help='Whether an item may select itself as an object-graph neighbour. '
                             'Control for the self-loop confound: the default graph has self '
                             'loops on 22.1% of rows, every tiebreak variant on 100%.')
    parser.add_argument('--knn_selfloop_alpha', type=float, default=0.0,
                        help='Add alpha*I to the object adjacency. The other half of the '
                             'self-loop control: adds the residual without changing the kNN.')
    parser.add_argument('--knn_sym', nargs='?', default='none', choices=['none', 'max', 'mean'],
                        help='Symmetrise every modality adjacency. max guarantees in-degree > 0.')

    # --- fusion ------------------------------------------------------------------------------
    parser.add_argument('--modalities', nargs='?', default='image,text,graph',
                        help='Modalities kept in the fusion; the rest get weight exactly 0.')
    parser.add_argument('--rho', type=float, default=1.0,
                        help='Gain on the graph-enhanced item representation when it is added to '
                             'the CF item embedding: x_hat = x_bar + rho * h/||h||  (paper Eq. 8, '
                             'which fixes it at 1 implicitly and never tunes it). rho=0 removes '
                             'the item graph entirely, leaving the bare CF model; rho>1 lets the '
                             'graph dominate. Note that ||h/||h|||| is exactly 1 while ||x_bar|| '
                             'is unconstrained and grows during training, so rho is NOT a '
                             'ratio -- the effective one is rho/||x_bar||, logged as '
                             'rho_effective in LATTICE_FUSION.')
    parser.add_argument('--freeze_fusion', type=int, default=0,
                        help='Hold the modality weights at their uniform initialisation instead '
                             'of learning them (1 = frozen). The ablation for "adaptive fusion": '
                             'modal_weight (and the per-item gate under --fusion gated|conf) get '
                             'requires_grad=False, while the per-modality projections *_trs stay '
                             'trainable, so only the WEIGHTING is ablated, not the modality '
                             'encoders. Measured motivation: over ~130 gradient steps the learned '
                             'weights move <0.01 from uniform, so this asks whether learning them '
                             'buys anything at all.')
    parser.add_argument('--lr_fusion', type=float, default=0.0,
                        help='Separate learning rate for modal_weight / gate / *_trs. 0 = use '
                             '--lr (published). These parameters only see a gradient on batch 0 '
                             'of each epoch (~130 steps a run), so at --lr they never leave '
                             'their initialization.')
    parser.add_argument('--content_path_beta', type=float, default=0.0,
                        help='Add beta*normalize(fused content feats) to the item embedding, so '
                             'modalities reach the score by more than item-graph topology.')
    parser.add_argument('--fast_laplacian', type=int, default=0,
                        help='Broadcast form of D^-1/2 A D^-1/2 (bit-identical, ~100x faster).')

    parser.add_argument('--n_layers', type=int, default=1,
                        help='Number of item graph conv layers')

    # ----------------------------------------------------------------------------------------
    # Item-graph propagation. The published operator is a parameter-free `item_adj @ h`, which
    # weights every neighbour by the fixed fused similarity. 'gat' replaces it with learned
    # attention over the *same* edges, so a neighbour can be down-weighted per item pair rather
    # than only per modality. 'dense' is the default, so every existing command line is
    # unchanged and takes the unchanged code path.
    # ----------------------------------------------------------------------------------------
    parser.add_argument('--item_prop', nargs='?', default='dense', choices=['dense', 'gat'],
                        help='Item-graph propagation {dense, gat}. dense is the published '
                             'parameter-free item_adj @ h.')
    parser.add_argument('--dump_embeddings', nargs='?', default='',
                        help='Path to save {ua, ia, epoch} whenever test_recall@20 improves, '
                             'i.e. the file always holds the best-epoch embeddings at exit. '
                             'Empty (default) does nothing -- no extra forward pass, no file.')
    parser.add_argument('--gat_heads', type=int, default=4,
                        help='Attention heads; concat, so embed_size must divide by this.')
    parser.add_argument('--gat_dropout', type=float, default=0.0,
                        help='Dropout on the attention coefficients.')
    parser.add_argument('--gat_act', nargs='?', default='none', choices=['none', 'elu'],
                        help='Post-aggregation nonlinearity. none matches the published linear op.')
    parser.add_argument('--gat_residual', type=int, default=0,
                        help='Add the input embedding back after aggregation.')
    parser.add_argument('--gat_init', nargs='?', default='reduce', choices=['reduce', 'xavier'],
                        help='reduce: W=I and a=0, so the layer starts as exactly the published '
                             'dense op and learns away from it. xavier: standard random init.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1, 0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='') 
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU id')
    parser.add_argument('--Ks', nargs='?', default='[10, 20]',
                        help='K value of ndcg/recall @ k')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')


    a = parser.parse_args()
    _check(parser, a)
    return a


def _check(parser, a):
    """Reject flag combinations where a flag would be silently ignored.

    A run whose knob does nothing looks exactly like a run whose knob did nothing, so every
    mode-specific flag must arrive with the mode that reads it.
    """
    if a.knn_tiebreak != 'none' and a.obj_knn == 'default':
        parser.error('--knn_tiebreak needs --obj_knn tiebreak|reserve|group')
    if a.obj_knn in ('tiebreak', 'group') and a.knn_tiebreak == 'none':
        parser.error('--obj_knn %s without --knn_tiebreak is a no-op' % a.obj_knn)
    if a.knn_tiebreak_eps < 0:
        parser.error('--knn_tiebreak_eps must be >= 0')
    if a.knn_tiebreak_eps and a.knn_tiebreak == 'none':
        parser.error('--knn_tiebreak_eps needs --knn_tiebreak')
    if a.knn_reserve and a.obj_knn != 'reserve':
        parser.error('--knn_reserve needs --obj_knn reserve')
    if a.obj_knn == 'reserve' and not a.knn_reserve:
        parser.error('--obj_knn reserve needs --knn_reserve > 0')
    if a.knn_group_m and a.obj_knn != 'group':
        parser.error('--knn_group_m needs --obj_knn group')
    if a.obj_knn == 'group' and a.knn_group_m < 0:
        parser.error('--knn_group_m must be >= 0')
    if a.knn_tb_seed and a.knn_tiebreak != 'random':
        parser.error('--knn_tb_seed needs --knn_tiebreak random')
    if a.knn_threshold and a.obj_knn != 'threshold':
        parser.error('--knn_threshold needs --obj_knn threshold')
    if a.obj_knn == 'threshold' and not 0 < a.knn_threshold <= 1:
        # 0 would keep every pair with a non-negative cosine, which is not a graph but a dense
        # block; the upper bound is where cosine tops out, so anything above it is an empty graph.
        parser.error('--obj_knn threshold needs 0 < --knn_threshold <= 1')
    mods = [m for m in a.modalities.split(',') if m]
    if not mods or any(m not in ('image', 'text', 'graph') for m in mods):
        parser.error('--modalities must be a non-empty subset of image,text,graph')
    if a.lr_fusion < 0 or a.content_path_beta < 0 or a.knn_selfloop_alpha < 0:
        parser.error('--lr_fusion / --content_path_beta / --knn_selfloop_alpha must be >= 0')
    # Same doctrine as above for the item-graph propagation knobs: a --gat_* that no code reads
    # produces a run indistinguishable from the control, which is the worst kind of null result.
    if a.item_prop != 'gat':
        for f, dflt in (('gat_heads', 4), ('gat_dropout', 0.0), ('gat_act', 'none'),
                        ('gat_residual', 0), ('gat_init', 'reduce')):
            if getattr(a, f) != dflt:
                parser.error('--%s needs --item_prop gat' % f)
    else:
        if a.gat_heads < 1 or a.embed_size % a.gat_heads:
            # concat=True splits embed_size across heads; a remainder would silently change the
            # output width and break the reduce-init identity.
            parser.error('--gat_heads must be >= 1 and divide --embed_size (%d)' % a.embed_size)
        if not 0 <= a.gat_dropout < 1:
            parser.error('--gat_dropout must be in [0, 1)')
