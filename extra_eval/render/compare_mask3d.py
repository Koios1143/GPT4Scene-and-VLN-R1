#!/usr/bin/env python3
"""Our Mask3D run (``mask3d_infer.py`` export) vs Chat-Scene's released ``mask3d_inst_seg``.

On ScanNet scenes both index ``_vh_clean_2`` vertices, so the first 100 proposals (what
GPT4Scene marks) can be compared directly: per rank, same class and mask IoU; and per released
proposal, the best mask IoU among ours (order-free). Checks that ``mask3d_infer.py`` reproduces
the pipeline behind GPT4Scene's marks before it is used on 3RScan / MultiScan.
"""

import argparse
import csv
import json
import os

import numpy as np
import torch

N = 100


def iou(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--export_dir", required=True)
    ap.add_argument("--released_dir", required=True, help="Chat-Scene mask3d_inst_seg/")
    ap.add_argument("--labels_tsv", required=True)
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    id2class = {}
    with open(args.labels_tsv) as f:
        rows = csv.reader(f, delimiter="\t")
        next(rows)
        for r in rows:
            id2class[r[0]] = r[2]
    res = {}
    for s in args.scenes.split(","):
        rel = torch.load(os.path.join(args.released_dir, f"{s}.pt"), map_location="cpu")[:N]
        ours = []
        for line in open(os.path.join(args.export_dir, f"{s}.txt")):
            path, cls, _ = line.split()
            ours.append((id2class[cls], np.nonzero(np.loadtxt(os.path.join(args.export_dir, path), dtype=np.uint8))[0]))
            if len(ours) == N:
                break
        n = min(len(rel), len(ours))
        rank_iou = [iou(rel[i]["segments"], ours[i][1]) for i in range(n)]
        same_cls = [rel[i]["label"] == ours[i][0] for i in range(n)]
        best = [max(iou(r["segments"], o[1]) for o in ours) for r in rel]
        res[s] = {"n_released": len(rel), "n_ours": len(ours),
                  "rank_mask_iou_mean": float(np.mean(rank_iou)), "rank_iou>0.9": float(np.mean(np.array(rank_iou) > 0.9)),
                  "rank_same_class": float(np.mean(same_cls)),
                  "best_match_iou_mean": float(np.mean(best)), "best_iou>0.9": float(np.mean(np.array(best) > 0.9))}
        print(s, json.dumps(res[s]))
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
