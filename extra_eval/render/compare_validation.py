#!/usr/bin/env python3
"""Released vs re-rendered GPT4Scene inputs on the same Beacon3D-ScanNet items.

Compares two ``score_beacon3d.py`` result dirs (``infer_grounding*.json`` with ``correct``,
``infer_qa.json`` and, when GPT-judged, ``infer_qa_processed.json``) on the items of
``--scenes`` that both contain: grounding case accuracy (both variants), QA EM / EM-R and
GPT-score, and how often the two runs agree item by item. Used to check that our renderer
(``render_marks.py``, rule ``released``) reproduces the released images closely enough for
its scores on scenes without released images to be comparable.
"""

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "evaluate"))
from utils.helper import answer_match, clean_answer  # noqa: E402


def load(d, scenes):
    out = {}
    for tag in ["", "_iou25"]:
        rows = json.load(open(os.path.join(d, f"infer_grounding{tag}.json")))
        out[f"grounding{tag}"] = {r["item_id"]: bool(r["correct"]) for r in rows if r["scan_id"] in scenes}
    qa = [r for r in json.load(open(os.path.join(d, "infer_qa.json"))) if r["scene_id"] in scenes]
    em = {r["question_id"]: answer_match(clean_answer(r["pred"]), [clean_answer(a) for a in r["answers"]]) for r in qa}
    out["qa_em"] = {k: v[0] for k, v in em.items()}
    out["qa_em_r"] = {k: v[1] for k, v in em.items()}
    proc = os.path.join(d, "infer_qa_processed.json")
    if os.path.exists(proc):
        by_q = {(r["scene_id"], r["question"].lower()): r["question_id"] for r in qa}
        out["qa_gpt"] = {by_q[k]: (it["score"] - 1) / 4 for it in json.load(open(proc))
                         if (k := (it["obj_id"].split("-")[0], it["question"])) in by_q}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--released", required=True, help="result dir of the run on the released images")
    ap.add_argument("--rendered", required=True, help="result dir of the run on our re-rendered images")
    ap.add_argument("--scenes", required=True, help="comma-separated validation scenes")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    scenes = set(args.scenes.split(","))
    a, b = load(args.released, scenes), load(args.rendered, scenes)
    res = {"scenes": sorted(scenes)}
    for k in a:
        if k not in b:
            continue
        keys = sorted(set(a[k]) & set(b[k]))
        x = np.array([a[k][i] for i in keys], float)
        y = np.array([b[k][i] for i in keys], float)
        res[k] = {"n": len(keys), "released": round(100 * x.mean(), 2), "rendered": round(100 * y.mean(), 2),
                  "diff": round(100 * (y.mean() - x.mean()), 2),
                  "item_agreement": round(100 * float(np.mean(x == y)), 2)}
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
