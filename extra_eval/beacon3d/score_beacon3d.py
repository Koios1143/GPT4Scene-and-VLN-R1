#!/usr/bin/env python3
"""GPT4Scene predictions on Beacon3D (ScanNet) -> Beacon3D infer files + official scores.

Reads ``<preds_dir>/qwenvl_beacon3d_scannet_{grounding,qa}/predictions_chunk_*.json``
(``evaluate/infer.py``), restricts Beacon3D's ``data/scannet`` + ``metadata`` to the scenes
GPT4Scene has released images for (``beacon3d_coverage.json`` from ``build_annos.py``) and
runs Beacon3D's evaluators on that subset, unmodified (grounding through
``beacon_grounding_shim.py``, QA with the GPT judge).

Grounding: GPT4Scene answers with the ``<OBJxxx>`` mark of a Mask3D proposal (its frames and
BEV image are marked with Mask3D instances, ``scannet_mask3d_val_attributes.pt``). As in
GPT4Scene's ScanRefer evaluation, the first mark below the scene's proposal count is the
prediction; no valid mark = wrong. The proposal box is matched to the ScanNet GT box
(``scannet_val_attributes.pt``, indexed by ScanNet objectId = Beacon3D ``target_id``) of
highest 3D IoU, and the case is correct when that GT object is a target and the IoU passes
the variant's threshold: ``""`` any overlap (closest to picking an object, as GT-mask models
do), ``"_iou25"`` IoU >= 0.25 (ScanRefer Acc@0.25 style). Same two variants as the
Grounded 3D-LLM evaluation in this project.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "evaluate"))
from utils.box_utils import box3d_iou, construct_bbox_corners  # noqa: E402
from utils.helper import answer_match, clean_answer  # noqa: E402

OBJ = re.compile(r"<OBJ(\d{3})>")
VARIANTS = {"": 0.0, "_iou25": 0.25}


def rel(text):
    return text.replace(REPO + "/", "")


def load_preds(preds_dir, task):
    paths = sorted(glob.glob(os.path.join(preds_dir, f"qwenvl_beacon3d_scannet_{task}", "predictions_chunk_*.json")))
    rows = [r for p in paths for r in json.load(open(p))]
    return {r["qid"]: r for r in rows}


def run_eval(name, cmd, beacon_root, out_dir):
    print(f"$ {' '.join(cmd)}", flush=True)
    p = subprocess.run(cmd, cwd=beacon_root, capture_output=True, text=True)
    with open(os.path.join(out_dir, f"score_{name}.txt"), "w") as f:
        f.write(rel(p.stdout + ("" if p.returncode == 0 else "\n[stderr]\n" + p.stderr)))
    print(p.stdout[-3000:])
    if p.returncode != 0:
        print(p.stderr[-3000:])
        raise SystemExit(f"{name}: evaluator failed")
    return p.stdout


def parse_grounding(txt):
    out = {k: float(v) for k, v in re.findall(r"\[(\w+)\]: ([\d.]+)", txt)}
    for k in ["chain", "object"]:
        out[k] = float(re.search(rf"^{k}: ([\d.]+)", txt, re.M).group(1))
    out["chain_four_types"] = [float(x) for x in re.search(r"four types: (.*)", txt).group(1).split(", ")]
    return out


def parse_qa(txt):
    m = re.search(r"EM = ([\d.]+) \| EM-R = ([\d.]+) \| GPT-Score = ([\d.]+)", txt)
    out = {"EM": float(m.group(1)), "EM-R": float(m.group(2)), "GPT-Score": float(m.group(3))}
    sc = txt.split("Scores:")[1].split("Chain analysis:")[0]
    out.update({k: float(v) for k, v in re.findall(r"^(\w+): ([\d.]+)", sc, re.M)})
    if "Chain analysis:" in txt:
        ch = txt.split("Chain analysis:")[1]
        out["chain_wo_extra"] = [float(x) for x in re.search(r"\[w/o extra\] four types: (.*)", ch).group(1).split(", ")]
        out["chain_overall"] = [float(x) for x in re.search(r"\[overall\] four types: (.*)", ch).group(1).split(", ")]
        out["R1"] = float(re.search(r"R1 \(higher better\): ([\d.]+)", ch).group(1))
        out["R2"] = float(re.search(r"R2 \(lower better\): ([\d.]+)", ch).group(1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preds_dir", required=True, help="evaluate/infer.py output_dir")
    ap.add_argument("--anno_dir", required=True, help="build_annos.py --out_dir (beacon3d_coverage.json)")
    ap.add_argument("--beacon_root", default=os.path.join(HERE, "Beacon3D"))
    ap.add_argument("--seg_attrs", default=os.path.join(REPO, "evaluate/annotation/scannet_mask3d_val_attributes.pt"))
    ap.add_argument("--gt_attrs", default=os.path.join(REPO, "evaluate/annotation/scannet_val_attributes.pt"))
    ap.add_argument("--result_dir", default=os.path.join(REPO, "extra_eval/results/beacon3d_scannet"))
    ap.add_argument("--qa_model", default="gpt-4o-mini-2024-07-18",
                    help="Beacon3D QA judge (upstream default gpt-4o-2024-11-20)")
    args = ap.parse_args()
    out_dir = os.path.abspath(args.result_dir)
    os.makedirs(out_dir, exist_ok=True)
    beacon_root = os.path.abspath(args.beacon_root)
    bdir = os.path.join(beacon_root, "data", "scannet")
    coverage = json.load(open(os.path.join(args.anno_dir, "beacon3d_coverage.json")))
    seg = torch.load(args.seg_attrs, map_location="cpu")
    gt = torch.load(args.gt_attrs, map_location="cpu")

    # Beacon3D data/metadata restricted to the covered scenes (evaluator inputs, not results)
    sub_dir = os.path.join(os.path.abspath(args.preds_dir), "beacon3d_subset")
    os.makedirs(sub_dir, exist_ok=True)
    src, sub = {}, {}
    for task, key in [("grounding", "scan_id"), ("qa", "scene_id")]:
        keep = set(coverage[task]["scenes_kept"])
        src[task] = [x for x in json.load(open(os.path.join(bdir, f"scannet_{task}.json"))) if x[key] in keep]
        meta = {s: v for s, v in json.load(open(os.path.join(bdir, f"metadata_scannet_{task}.json"))).items() if s in keep}
        sub[task] = (os.path.join(sub_dir, f"scannet_{task}.json"), os.path.join(sub_dir, f"metadata_scannet_{task}.json"))
        json.dump(src[task], open(sub[task][0], "w"))
        json.dump(meta, open(sub[task][1], "w"))
    all_idx = {t: [i for i, x in enumerate(json.load(open(os.path.join(bdir, f"scannet_{t}.json"))))
                   if x["scan_id" if t == "grounding" else "scene_id"] in set(coverage[t]["scenes_kept"])]
               for t in ["grounding", "qa"]}

    res = {"benchmark": "Beacon3D ScanNet", "coverage": {
        t: {k: coverage[t][k] for k in ["items", "items_kept"]} | {"scenes_kept": len(coverage[t]["scenes_kept"]),
             "scenes_dropped": coverage[t]["scenes_dropped"]} for t in coverage}}

    # ---- grounding ----
    preds = load_preds(args.preds_dir, "grounding")
    assert sorted(preds) == all_idx["grounding"], f"grounding: {len(preds)} preds for {len(all_idx['grounding'])} items"
    rows, no_mark = [], 0
    for i, s in zip(all_idx["grounding"], src["grounding"]):
        p = preds[i]
        scene = s["scan_id"]
        n_prop = seg[scene]["locs"].shape[0]
        pid = next((int(m.group(1)) for m in OBJ.finditer(p["pred"]) if int(m.group(1)) < n_prop), None)
        obj, iou = None, 0.0
        if pid is None:
            no_mark += 1
        else:
            pl = seg[scene]["locs"][pid].tolist()
            pc = construct_bbox_corners(pl[:3], pl[3:])
            ious = [box3d_iou(pc, construct_bbox_corners(g[:3], g[3:])) for g in gt[scene]["locs"].tolist()]
            obj = int(np.argmax(ious))
            iou = float(ious[obj])
        rows.append({"item_id": s["item_id"], "scan_id": scene, "pred": p["pred"], "pred_proposal": pid,
                     "pred_object_id": obj, "pred_iou": iou, "target_id": s["target_id"]})
    res["grounding_no_valid_mark"] = no_mark
    for tag, thr in VARIANTS.items():
        infer = [r | {"correct": r["pred_object_id"] in r["target_id"]
                      and (r["pred_iou"] >= thr if thr else r["pred_iou"] > 0)} for r in rows]
        path = os.path.join(out_dir, f"infer_grounding{tag}.json")
        json.dump(infer, open(path, "w"), indent=1)
        txt = run_eval(f"grounding{tag}", [sys.executable, os.path.join(HERE, "beacon_grounding_shim.py"),
                                           "--infer", path, "--data", sub["grounding"][0],
                                           "--metadata", sub["grounding"][1]], beacon_root, out_dir)
        res[f"grounding{tag or '_best'}"] = parse_grounding(txt)

    # ---- QA ----
    preds = load_preds(args.preds_dir, "qa")
    assert sorted(preds) == all_idx["qa"], f"qa: {len(preds)} preds for {len(all_idx['qa'])} items"
    infer = [{"question_id": s["question_id"], "scene_id": s["scene_id"], "question": s["question"],
              "pred": preds[i]["pred"], "answers": s["answers"]} for i, s in zip(all_idx["qa"], src["qa"])]
    path = os.path.join(out_dir, "infer_qa.json")
    json.dump(infer, open(path, "w"), indent=1)
    em = [answer_match(clean_answer(r["pred"]), [clean_answer(a) for a in r["answers"]]) for r in infer]
    res["qa_em_local"] = {"EM": 100 * np.mean([e[0] for e in em]), "EM-R": 100 * np.mean([e[1] for e in em]),
                          "empty_preds": sum(not r["pred"].strip() for r in infer)}
    if not os.environ.get("OPENAI_API_KEY"):
        print("qa: OPENAI_API_KEY unset, GPT-judged evaluation skipped")
    else:
        res["qa_judge"] = args.qa_model
        for tag in VARIANTS:  # the grounding->QA chain analysis needs a grounding variant
            processed = os.path.join(out_dir, f"infer_qa_processed{tag}.json")
            if tag == "":
                if os.path.exists(processed):
                    os.remove(processed)  # evaluate_qa.py silently reuses an existing processed file
            else:  # the judge's scores do not depend on grounding: reuse them instead of re-judging
                json.dump(json.load(open(os.path.join(out_dir, "infer_qa_processed.json"))), open(processed, "w"), indent=1)
            txt = run_eval(f"qa{tag}", [sys.executable, "evaluate_qa.py", "--infer", path,
                                        "--data", sub["qa"][0], "--metadata", sub["qa"][1],
                                        "--model", args.qa_model, "--output", processed,
                                        "--grounding", os.path.join(out_dir, f"infer_grounding{tag}_processed.json")],
                           beacon_root, out_dir)
            res[f"qa{tag or '_best'}"] = parse_qa(txt)
        os.remove(os.path.join(out_dir, "infer_qa_processed_iou25.json"))
    json.dump(res, open(os.path.join(out_dir, "scores.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
