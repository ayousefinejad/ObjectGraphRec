# coding: utf-8
"""One-config diagnostic runner for CRANE.

    python main_diag.py -d home_v2 --dropout 0.9 --reg 0.001 --batch_first 1

main.py runs the full `hyper_parameters` grid from configs/model/CRANE.yaml (8 configs, ~2.3 h on
home_v2). This pins dropout and reg_weight to a single value each so the grid is 1 config, and
exposes `cross_modal_batch_first` so the published (False) and intended (True) cross-modal
attention semantics can be compared at an otherwise identical setting.

config_dict wins over both yaml files -- utils/configurator.py:64 applies it last -- so these
override CRANE.yaml without editing it. No model code is changed by this file.
"""

import os
import argparse
from utils.quick_start import quick_start

os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='CRANE')
    parser.add_argument('--dataset', '-d', type=str, default='home_v2')
    # Omit either to keep CRANE.yaml's full list for that axis, i.e. run the published 8-config
    # grid rather than one point. That is what makes a --batch_first 1 run comparable to the
    # published one: both are then a best-of-8 selected the same way.
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--reg', type=float)
    parser.add_argument('--batch_first', type=int, default=0, choices=(0, 1))
    parser.add_argument('--object', type=int, default=0, choices=(0, 1),
                        help='object graph before the latent feature learning module')
    parser.add_argument('--object_in_attention', type=int, default=1, choices=(0, 1),
                        help='0 = graph path only, for ablating the attention token')
    parser.add_argument('--n_obj_layers', type=int, default=1)
    parser.add_argument('--epochs', type=int, help='cap epochs, e.g. 2 for a smoke test')
    args, _ = parser.parse_known_args()

    # Each must stay a list: quick_start builds the grid with itertools.product over
    # config[p] for p in hyper_parameters, so a bare float would be iterated character-wise.
    config_dict = {'gpu_id': 0, 'cross_modal_batch_first': bool(args.batch_first),
                   'use_object_graph': bool(args.object),
                   'object_in_attention': bool(args.object_in_attention),
                   'n_obj_layers': args.n_obj_layers}
    if args.epochs is not None:
        config_dict['epochs'] = args.epochs
    if args.dropout is not None:
        config_dict['dropout'] = [args.dropout]
    if args.reg is not None:
        config_dict['reg_weight'] = [args.reg]

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=False)
