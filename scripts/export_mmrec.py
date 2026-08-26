#!/usr/bin/env python3
"""Export home_v2-2 from LATTICE layout into the MMRec layout CRANE expects.

    python scripts/export_mmrec.py                    # writes CRANE/CRANE/data/home_v2/
    python scripts/export_mmrec.py --check            # verify an existing export, write nothing

LATTICE stores the splits as three user -> [item] dicts (5-core/{train,val,test}.json) and reads
features straight out of the dataset root. MMRec wants ONE interaction table with an integer split
label per row (utils/dataset.py:split filters on `x_label` == 0/1/2) plus the same .npy features.
So this is a pure reshape: identical interactions, identical item indexing, nothing recomputed.

Nothing under data/home_v2-2/ is written or moved. The two feature files are symlinked, not copied
-- image_feat.npy alone is 475 MB and CRANE only ever np.loads it read-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "home_v2-2"
DST = ROOT.parent / "CRANE" / "CRANE" / "data" / "home_v2"
CFG = ROOT.parent / "CRANE" / "CRANE" / "src" / "configs" / "dataset" / "home_v2.yaml"

# The object features, for CRANE's ObjectGraph. Taken from the `default_fixed` variant rather
# than data/home_v2-2/object_feat.npy -- the two differ, and default_fixed is the SAGE/MIT+NYU
# encoder the LATTICE obj_lgn3 runs used, so this is the arm CRANE's object graph is comparable
# to. Symlinked read-only like the other two feature files.
OBJ_SRC = ROOT / "data" / "lattice-runs" / "default_fixed" / "object_feat.npy"

# MMRec's split label: utils/dataset.py iterates `for i in range(3)` over this column.
SPLITS = {"train": 0, "val": 1, "test": 2}

# The field names must match configs/dataset/home_v2.yaml, which overrides the `user_id:token`
# style in configs/overall.yaml -- baby.yaml does the same. dataset.py reads exactly these three
# columns via usecols, so a timestamp column would be loaded and discarded; it is not written.
HEADER = "userID\titemID\tx_label"

YAML = """# home_v2 (65,139 users x 14,503 items), exported from the LATTICE layout by
# scripts/export_mmrec.py in the object-graph repo. Interactions are byte-identical to
# data/home_v2-2/5-core/{train,val,test}.json; only the container changed.
USER_ID_FIELD: userID
ITEM_ID_FIELD: itemID

filter_out_cod_start_users: True

inter_file_name: 'home_v2.inter'

# name of features
vision_feature_file: 'image_feat.npy'
text_feature_file: 'text_feat.npy'

field_separator: "\\t"
"""


def load_splits() -> dict[str, dict]:
    return {name: json.loads((SRC / "5-core" / f"{name}.json").read_text()) for name in SPLITS}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="verify an existing export, write nothing")
    a = p.parse_args()

    splits = load_splits()
    counts = {k: sum(len(v) for v in d.values()) for k, d in splits.items()}

    n_items = np.load(SRC / "image_feat.npy", mmap_mode="r").shape[0]
    text_items = np.load(SRC / "text_feat.npy", mmap_mode="r").shape[0]
    obj_items = np.load(OBJ_SRC, mmap_mode="r").shape[0]
    max_item = max(i for d in splits.values() for items in d.values() for i in items)
    # CRANE indexes nn.Embedding.from_pretrained(v_feat) by raw item id, so a feature matrix with
    # fewer rows than max_item+1 is an out-of-bounds read at the first forward, not a warning.
    assert n_items == text_items, f"feature row mismatch: image {n_items} vs text {text_items}"
    assert n_items == obj_items, f"feature row mismatch: image {n_items} vs object {obj_items}"
    assert max_item < n_items, f"item id {max_item} exceeds {n_items} feature rows"

    if a.check:
        inter = DST / "home_v2.inter"
        if not inter.exists():
            raise SystemExit(f"missing {inter} -- run without --check first")
        lines = inter.read_text().splitlines()
        got: dict[int, int] = {}
        for line in lines[1:]:
            got[int(line.rsplit("\t", 1)[1])] = got.get(int(line.rsplit("\t", 1)[1]), 0) + 1
        print(f"{inter.relative_to(ROOT.parent)}: {len(lines) - 1} rows")
        for name, label in SPLITS.items():
            ok = "OK" if got.get(label) == counts[name] else "MISMATCH"
            print(f"  x_label={label} ({name}): {got.get(label)} vs {counts[name]} expected  {ok}")
        return

    DST.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, label in SPLITS.items():
        for user, items in splits[name].items():
            for item in items:
                rows.append(f"{int(user)}\t{int(item)}\t{label}")
    (DST / "home_v2.inter").write_text(HEADER + "\n" + "\n".join(rows) + "\n")

    for feat, src in (("image_feat.npy", SRC / "image_feat.npy"),
                      ("text_feat.npy", SRC / "text_feat.npy"),
                      ("object_feat.npy", OBJ_SRC)):
        link = DST / feat
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(src.resolve())

    CFG.write_text(YAML)

    n_users = max(int(u) for d in splits.values() for u in d) + 1
    print(f"wrote {DST.relative_to(ROOT.parent)}/home_v2.inter: {len(rows)} rows "
          f"({', '.join(f'{k}={v}' for k, v in counts.items())})")
    print(f"      n_users={n_users} n_items={n_items} (features symlinked, not copied)")
    print(f"wrote {CFG.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    main()
