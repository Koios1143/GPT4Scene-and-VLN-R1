#!/usr/bin/env python3
"""Which Mask3D proposals get an ``<OBJxxx>`` mark, per scene, under three rules.

GPT4Scene's marks are Mask3D proposals (Chat-Scene's ``mask3d_inst_seg``, the first 100 per
scene, whose boxes are exactly ``scannet_mask3d_val_attributes.pt``). Its released images do
not mark all 100. Reading the ids off the released BEV images of scene0011_00 / 0025_00 /
0164_00, the marked set is (70 / 70 released ids recovered, 2 extra):
the best box-IoU proposal of every GT object referenced in ScanRefer / Multi3DRefer val, plus
the best proposal of every other non-structure GT object with box IoU >= 0.8 -- i.e. it uses GT
instances and another benchmark's annotations. Rules written to ``<out>``:

* ``released`` -- that reconstruction of the released rule. Only to validate our renderer
  against the released images on scenes that have them (scenes without ScanRefer annotations
  get only the IoU >= 0.8 part).
* ``gt`` -- every GT object except wall / floor / ceiling gets its best box-IoU proposal
  (IoU >= 0.25). Annotation-agnostic but uses GT instances, like a GT-mask model (Chat-Scene).
* ``all`` -- every proposal, no GT: all 100 minus predicted wall / floor / ceiling and minus
  near-duplicate boxes (IoU >= 0.95 with a lower id), like a detector-only model.

Box IoU = GPT4Scene's own ``box3d_iou`` on (center, size) boxes.
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "evaluate"))
from utils.box_utils import box3d_iou, construct_bbox_corners  # noqa: E402

STRUCTURE = {"wall", "floor", "ceiling"}
GT_IOU = 0.25
RELEASED_IOU = 0.8
DUP_IOU = 0.95


def corners(locs):
    return [construct_bbox_corners(l[:3], l[3:]) for l in locs.tolist()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--anno_dir", default=os.path.join(REPO, "evaluate/annotation"), help="GPT4Scene-Val-Annotation")
    ap.add_argument("--scenes", required=True, help="comma-separated scene ids")
    ap.add_argument("--out", required=True, help="output json {rule: {scene: [proposal ids]}}")
    args = ap.parse_args()
    seg = torch.load(os.path.join(args.anno_dir, "scannet_mask3d_val_attributes.pt"), map_location="cpu")
    gt = torch.load(os.path.join(args.anno_dir, "scannet_val_attributes.pt"), map_location="cpu")
    scanrefer = json.load(open(os.path.join(args.anno_dir, "scanrefer/ScanRefer_filtered.json")))
    multi3dref = json.load(open(os.path.join(args.anno_dir, "multi3dref_mask3d_val.json")))
    rules = {"released": {}, "gt": {}, "all": {}}
    for s in args.scenes.split(","):
        P, G = corners(seg[s]["locs"]), corners(gt[s]["locs"])
        iou = np.array([[box3d_iou(p, g) for g in G] for p in P])  # (n_prop, n_gt)
        best = iou.argmax(0)
        referenced = {int(x["object_id"]) for x in scanrefer if x["scene_id"] == s}
        referenced |= {g for x in multi3dref if x["scene_id"] == s for g in x["ref_captions"]}
        structure = [name in STRUCTURE for name in gt[s]["objects"]]
        rules["released"][s] = sorted({int(best[g]) for g in referenced if iou[best[g], g] > 0}
                                      | {int(best[g]) for g in range(len(G))
                                         if not structure[g] and iou[best[g], g] >= RELEASED_IOU})
        rules["gt"][s] = sorted({int(best[g]) for g in range(len(G)) if not structure[g] and iou[best[g], g] >= GT_IOU})
        keep = []
        for i, name in enumerate(seg[s]["objects"][:len(P)]):  # objects lists every instance, locs the first 100
            if name in STRUCTURE or not np.any(seg[s]["locs"][i, 3:].numpy() > 0):
                continue
            if any(box3d_iou(P[i], P[j]) >= DUP_IOU for j in keep):
                continue
            keep.append(i)
        rules["all"][s] = keep
        print(s, {k: len(v[s]) for k, v in rules.items()})
    json.dump(rules, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
