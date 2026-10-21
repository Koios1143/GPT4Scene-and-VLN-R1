#!/usr/bin/env python3
"""Mask3D export (``mask3d_infer.py``) -> GPT4Scene-style proposals for 3RScan / MultiScan.

Same conversion as Chat-Scene's ``preprocess/prepare_mask3d_data.py`` (which produced the
ScanNet ``mask3d_inst_seg`` GPT4Scene's marks come from): every exported instance, in export
order (descending score), as ``{"label": <ScanNet200 category>, "segments": <vertex ids>}``;
labels via ``scannetv2-labels.combined.tsv`` (id -> category). GPT4Scene uses the first 100.
Writes under ``--out_dir``:

* ``inst_seg/<scene>.pt`` -- that list (vertex ids index ``prepare_scene_rgbd.py``'s mesh);
* ``seg_attrs.pt`` / ``gt_attrs.pt`` -- ``{scene: {"objects", "locs"}}`` like GPT4Scene's
  ``scannet_{mask3d_,}val_attributes.pt``: first-100 proposal boxes / GT object boxes
  (center + size, z-up frame); GT ``locs`` is indexed by objectId (zeros for unused ids) so
  ``score_beacon3d.py`` can match proposal -> GT by box IoU exactly as on ScanNet;
* ``mark_sets.json`` -- rule ``all`` (``mark_sets.all_rule``).
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mark_sets import all_rule  # noqa: E402

MAX_PROPOSALS = 100


def box(p):
    return np.r_[(p.max(0) + p.min(0)) / 2, p.max(0) - p.min(0)] if len(p) else np.zeros(6)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--export_dir", required=True, help=".../instance_evaluation_<name>_0/decoder_-1")
    ap.add_argument("--scene_dir", required=True, help="prepare_scene_rgbd.py --out_dir")
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--labels_tsv", required=True, help="scannetv2-labels.combined.tsv")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    id2class = {}
    with open(args.labels_tsv) as f:
        rows = csv.reader(f, delimiter="\t")
        next(rows)
        for r in rows:
            id2class[r[0]] = r[2]
    os.makedirs(os.path.join(args.out_dir, "inst_seg"), exist_ok=True)
    seg_attrs, gt_attrs, marks = {}, {}, {"all": {}}
    for s in args.scenes.split(","):
        sc = np.load(os.path.join(args.scene_dir, f"{s}.npz"))
        xyz = (np.c_[sc["xyz"], np.ones(len(sc["xyz"]))] @ sc["align"].T)[:, :3]
        insts = []
        for line in open(os.path.join(args.export_dir, f"{s}.txt")):
            path, cls, _score = line.split()
            mask = np.loadtxt(os.path.join(args.export_dir, path), dtype=np.uint8)
            insts.append({"label": id2class[cls], "segments": np.nonzero(mask)[0].tolist()})
        torch.save(insts, os.path.join(args.out_dir, "inst_seg", f"{s}.pt"))
        locs = np.stack([box(xyz[it["segments"]]) for it in insts[:MAX_PROPOSALS]])
        seg_attrs[s] = {"objects": [it["label"] for it in insts], "locs": torch.tensor(locs, dtype=torch.float32)}
        objects = {int(k): v for k, v in json.loads(str(sc["objects"])).items()}
        n = max(objects) + 1
        gl = np.zeros((n, 6))
        for o in objects:
            gl[o] = box(xyz[sc["vert_obj"] == o])
        gt_attrs[s] = {"objects": [objects.get(i, "") for i in range(n)], "locs": torch.tensor(gl, dtype=torch.float32)}
        marks["all"][s] = all_rule(locs, seg_attrs[s]["objects"])
        print(s, f"{len(insts)} instances, {len(marks['all'][s])} marked (rule all)", flush=True)
    torch.save(seg_attrs, os.path.join(args.out_dir, "seg_attrs.pt"))
    torch.save(gt_attrs, os.path.join(args.out_dir, "gt_attrs.pt"))
    json.dump(marks, open(os.path.join(args.out_dir, "mark_sets.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
