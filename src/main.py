import datetime
import json
import math
import os
import random
import sys
from time import time
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.sparse as sparse

from utility.parser import parse_args
from Models import LATTICE
from utility.batch_test import *

args = parse_args()


class Trainer(object):
    def __init__(self, data_config):
        # argument settings
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.model_name = args.model_name
        self.mess_dropout = eval(args.mess_dropout)
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.weight_size = eval(args.weight_size)
        self.n_layers = len(self.weight_size)
        self.regs = eval(args.regs)
        self.decay = self.regs[0]

        self.norm_adj = data_config['norm_adj']
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().cuda()
        
        image_feats = np.load('data/{}/image_feat.npy'.format(args.dataset))
        text_feats = np.load('data/{}/text_feat.npy'.format(args.dataset))
        graph_feats = np.load('data/{}/object_feat.npy'.format(args.dataset))

        self.model = LATTICE(self.n_users, self.n_items, self.emb_dim, self.weight_size, self.mess_dropout, image_feats, text_feats, graph_feats)
        self.model = self.model.cuda()
        # The fusion parameters see a gradient on batch 0 of each epoch only: `build_item_graph`
        # is set True once per epoch and `item_adj` is detached for the other 149 batches, so
        # over a ~130-epoch run modal_weight gets ~130 Adam steps. At lr 5e-4 that moves the
        # logits by ~0.06 and the softmax weights stay within ~0.01 of uniform -- the "learned
        # modality weights" never leave their initialization. --lr_fusion gives them their own
        # step size; 0 falls through to --lr, which is the published optimizer exactly.
        fusion_pfx = ('modal_weight', 'gate.', 'image_trs.', 'text_trs.', 'graph_trs.')
        named = list(self.model.named_parameters())
        fusion_p = [p for n, p in named if n.startswith(fusion_pfx)]
        other_p = [p for n, p in named if not n.startswith(fusion_pfx)]
        assert len(fusion_p) + len(other_p) == len(named)
        if args.lr_fusion > 0:
            print('LATTICE_OPT lr_fusion=%g on %d tensors (%d others at lr=%g)'
                  % (args.lr_fusion, len(fusion_p), len(other_p), self.lr), flush=True)
            self.optimizer = optim.Adam([{'params': other_p, 'lr': self.lr},
                                         {'params': fusion_p, 'lr': args.lr_fusion}], lr=self.lr)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = self.set_lr_scheduler()

    def set_lr_scheduler(self):
        fac = lambda epoch: 0.96 ** (epoch / 50)
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        return scheduler

    def test(self, users_to_test, is_val):
        self.model.eval()
        with torch.no_grad():
            ua_embeddings, ia_embeddings = self.model(self.norm_adj, build_item_graph=True)
        result = test_torch(ua_embeddings, ia_embeddings, users_to_test, is_val)
        # Stashed rather than returned, so every existing call site (and its return-value
        # unpacking) is untouched. Only the final test() call before a best-epoch update is read.
        self._last_ua, self._last_ia = ua_embeddings, ia_embeddings
        return result

    def train(self):
        training_time_list = []
        loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
        stopping_step = 0
        should_stop = False
        cur_best_pre_0 = 0.

        n_batch = data_generator.n_train // args.batch_size + 1
        best_recall = 0
        for epoch in (range(args.epoch)):
            t1 = time()
            loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
            n_batch = data_generator.n_train // args.batch_size + 1
            f_time, b_time, loss_time, opt_time, clip_time, emb_time = 0., 0., 0., 0., 0., 0.
            sample_time = 0.
            build_item_graph = True
            for idx in (range(n_batch)):
                self.model.train()
                self.optimizer.zero_grad()
                sample_t1 = time()
                users, pos_items, neg_items = data_generator.sample()
                sample_time += time() - sample_t1                                                 
                ua_embeddings, ia_embeddings = self.model(self.norm_adj, build_item_graph=build_item_graph)
                build_item_graph = False
                u_g_embeddings = ua_embeddings[users]
                pos_i_g_embeddings = ia_embeddings[pos_items]
                neg_i_g_embeddings = ia_embeddings[neg_items]


                batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                              neg_i_g_embeddings)

                batch_loss = batch_mf_loss + batch_emb_loss + batch_reg_loss

                batch_loss.backward(retain_graph=True)
                if idx == 0:
                    # Batch 0 is the only batch where item_adj is rebuilt, so it is the only one
                    # where the fusion parameters have a gradient at all (every later batch takes
                    # the detach() branch). zero_grad(set_to_none=True) at the top of the loop
                    # means reading .grad anywhere else reports None and would be misleading.
                    #
                    # This exists because a fusion path that has been silently severed looks
                    # exactly like a fusion that had nothing to learn. Recording the norm makes
                    # a 0.0 greppable in every future run of every arm instead of invisible.
                    self._grad_b0 = {}
                    for _n, _p in self.model.named_parameters():
                        if (_n in ('modal_weight', 'image_trs.weight', 'text_trs.weight',
                                   'graph_trs.weight', 'gate.weight')
                                or _n.startswith('item_gat')):
                            # 3 significant figures, not round(., 6): the object projection's
                            # gradient is legitimately ~1e-7 while modal_weight's is ~1e-4, and
                            # fixed-decimal rounding would print both a severed path and a small
                            # one as "0.0" -- collapsing exactly the distinction this records.
                            self._grad_b0[_n.replace('.weight', '') + '_grad_b0'] = (
                                0.0 if _p.grad is None else float('%.3g' % _p.grad.norm()))
                self.optimizer.step()

                loss += float(batch_loss)
                mf_loss += float(batch_mf_loss)
                emb_loss += float(batch_emb_loss)
                reg_loss += float(batch_reg_loss)


            self.lr_scheduler.step()

            del ua_embeddings, ia_embeddings, u_g_embeddings, neg_i_g_embeddings, pos_i_g_embeddings

            if math.isnan(loss) == True:
                print('ERROR: loss is nan.')
                # Bare sys.exit() returns 0, so the study runner's `proc.returncode != 0` check
                # never saw this as a failure -- it parsed whatever `test==` line had already
                # printed (from before the collapse) and recorded a NaN run as a real score.
                sys.exit(1)

            perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                epoch, time() - t1, loss, mf_loss, emb_loss)
            training_time_list.append(time() - t1)
            print(perf_str)

            if epoch % args.verbose != 0:
                continue


            t2 = time()
            users_to_test = list(data_generator.test_set.keys())
            users_to_val = list(data_generator.val_set.keys())
            ret = self.test(users_to_val, is_val=True)
            # A flat --lr_fusion arm is ambiguous without this: weights that moved and found
            # uniform to be near-optimal look identical, in the metrics, to weights that never
            # moved at all. Printed at every eval, so the epoch-0 row is the baseline.
            print('LATTICE_FUSION %d %s' % (
                epoch, json.dumps({**self.model.fusion_state(),
                                   **getattr(self, '_grad_b0', {})})), flush=True)
            training_time_list.append(t2 - t1)

            t3 = time()

            loss_loger.append(loss)
            rec_loger.append(ret['recall'])
            pre_loger.append(ret['precision'])
            ndcg_loger.append(ret['ndcg'])
            hit_loger.append(ret['hit_ratio'])
            if args.verbose > 0:
                perf_str = 'Epoch %d [%.1fs + %.1fs]:  val==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f], ' \
                           'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                           (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss, ret['recall'][0],
                            ret['recall'][-1],
                            ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                            ret['ndcg'][0], ret['ndcg'][-1])
                print(perf_str)

            if ret['recall'][1] > best_recall:
                best_recall = ret['recall'][1]
                test_ret = self.test(users_to_test, is_val=False)
                perf_str = 'Epoch %d [%.1fs + %.1fs]: test==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f], ' \
                           'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                           (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss, test_ret['recall'][0],
                            test_ret['recall'][-1],
                            test_ret['precision'][0], test_ret['precision'][-1], test_ret['hit_ratio'][0], test_ret['hit_ratio'][-1],
                            test_ret['ndcg'][0], test_ret['ndcg'][-1])
                print(perf_str)
                if args.dump_embeddings:
                    # self._last_ua/_last_ia are from the test_ret call two lines up (is_val=False),
                    # i.e. exactly the embeddings that produced the numbers just printed above --
                    # not a fresh forward pass that could drift from them under dropout/BN.
                    torch.save({'ua': self._last_ua.detach().cpu(),
                               'ia': self._last_ia.detach().cpu(),
                               'epoch': epoch, 'item_prop': args.item_prop,
                               'dataset': args.dataset, 'seed': args.seed},
                              args.dump_embeddings)
                stopping_step = 0
            elif stopping_step < args.early_stopping_patience:
                stopping_step += 1
                print('#####Early stopping steps: %d #####' % stopping_step)
            else:
                print('#####Early stop! #####')
                break

        print(test_ret)

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1./2*(users**2).sum() + 1./2*(pos_items**2).sum() + 1./2*(neg_items**2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.decay * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu

if __name__ == '__main__':
    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()
    config['norm_adj'] = norm_adj

    trainer = Trainer(data_config=config)
    trainer.train()

