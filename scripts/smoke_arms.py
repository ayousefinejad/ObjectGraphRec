#!/usr/bin/env python3
"""Print the LATTICE_DIAG line for each named arm without training it.

    python scripts/smoke_arms.py tb_text_nosf selfloop res2_text grp3_text sym_max
    python scripts/smoke_arms.py tuned:thr08          # variant:arm, default variant default_fixed

The point is to prove, before spending ~23 min/run on the screen, that every arm's flags actually
reach the graph: an arm whose diagnostics equal the control's has a silently-ignored flag, a stale
cache, or the wrong dataset directory.

`--epoch 0` skips the training loop entirely, so the run ends in a NameError on main.py's final
`print(test_ret)` -- expected, and it lands *after* the model (and therefore the diagnostics) is
built. Overrides go after BASE, because argparse takes the last occurrence and BASE ends with
`--epoch 200`.
"""

from __future__ import annotations

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_lattice_study import ARMS, BASE, arm_dataset, stale_caches  # noqa: E402

KEEP = ('LATTICE_DIAG', 'LATTICE_KNN', 'LATTICE_WARN', 'LATTICE_OPT')


def smoke(spec: str) -> None:
    variant, _, arm = spec.rpartition(':')
    variant = variant or 'default_fixed'
    ds = arm_dataset(variant, arm, stale_caches(ARMS[arm]))
    # A real (non-symlink) cache in an arm directory is a leftover from an earlier run of a
    # *different* arm; `load_or_build_adj` would hard-error on it, but for a smoke test we just
    # want it gone. Symlinked caches are the shared image/text ones and must survive.
    for m in ('image', 'text', 'graph'):
        p = f"data/{ds}/5-core/{m}_adj_10.pt"
        if os.path.exists(p) and not os.path.islink(p):
            os.remove(p)
    cmd = [sys.executable, 'main.py', *BASE, *ARMS[arm],
           '--dataset', ds, '--seed', '0', '--epoch', '0', '--fast_laplacian', '1', '--verbose', '100']
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(f"=== {variant}:{arm} ===", flush=True)
    for l in (r.stdout + r.stderr).splitlines():
        if l.startswith(KEEP) or ('Error' in l and 'test_ret' not in l):
            print(l, flush=True)


if __name__ == '__main__':
    for a in sys.argv[1:]:
        smoke(a)
