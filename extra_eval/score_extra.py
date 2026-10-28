#!/usr/bin/env python3
"""Score ScanRefer / Multi3DRefer / Scan2Cap / ScanQA / SQA3D / Real-3DQA predictions of
``run_infer.sh`` (either path). CPU only; pycocoevalcap needs Java.

* ScanRefer, Multi3DRefer, Scan2Cap, ScanQA, SQA3D (val = GPT4Scene's paper split, and test):
  the repo's own scorers in ``evaluate/utils/eval.py``, as ``evaluate/calculate_scores.py``
  calls them (Mask3D attributes ``annotation/scannet_mask3d_val_attributes.pt``).
* Real-3DQA: the official ``evaluate_debiased_exact_match.py`` (EM, refined EM) on
  ``debiased_test`` and on each rotation split, and ``evaluate_rotation_robustness.py``
  (One..Four, VRS) on the four rotation splits. An empty generation is replaced by
  ``<empty>`` first; otherwise the refined match counts it as correct, because the empty
  string is a substring of every answer.

Predictions are merged over ``qwenvl_<task>/predictions_chunk_*.json`` and checked against
the annotation file (one prediction per qid). Writes ``<result_dir>/<bench>/scores.json`` +
``preds.json`` (+ Real-3DQA prediction files) and ``<result_dir>/summary.json``.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "evaluate"))
sys.path.insert(0, str(HERE / "real3dqa"))

REAL3DQA_SPLITS = ["debiased_test", "rotation_0", "rotation_90", "rotation_180", "rotation_270"]
ROTATIONS = REAL3DQA_SPLITS[1:]
EXPECTED = {"scanrefer": 9508, "multi3dref": 11120, "scan2cap": 2007, "scanqa": 4675, "sqa3d": 3261,
            "sqa3d_test": 3519, "real3dqa_debiased_test": 1485, "real3dqa_rotation_0": 750,
            "real3dqa_rotation_90": 750, "real3dqa_rotation_180": 750, "real3dqa_rotation_270": 750}
EMPTY = "<empty>"


def load_preds(pred_dir, task, anno_file):
    files = sorted(glob.glob(os.path.join(pred_dir, f"qwenvl_{task}", "predictions_chunk_*.json")))
    if not files:
        return None
    by_qid = {}
    for f in files:
        for p in json.load(open(f)):
            by_qid.setdefault(p["qid"], p)
    annos = json.load(open(anno_file))
    qids = [a.get("qid", 0) for a in annos]
    missing = [q for q in qids if q not in by_qid]
    assert not missing, f"{task}: {len(missing)} annotation items have no prediction"
    assert len(qids) == EXPECTED[task], f"{task}: expected {EXPECTED[task]}, annotation has {len(qids)}"
    return [by_qid[q] for q in qids]


def coco_scorers():
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    # as in evaluate/calculate_scores.py
    return PTBTokenizer(), [(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]), (Meteor(), "METEOR"),
                            (Rouge(), "ROUGE_L"), (Cider(), "CIDEr")]


def score_upstream(task, preds):
    from utils import eval as ev
    config = SimpleNamespace(seg_val_attr_file="annotation/scannet_mask3d_val_attributes.pt", base_model=False)
    if task in ("scan2cap", "scanqa", "sqa3d", "sqa3d_test"):
        tokenizer, scorers = coco_scorers()
    if task == "scanrefer":
        s = ev.calc_scanrefer_score(preds, config)
    elif task == "multi3dref":
        s = ev.calc_multi3dref_score(preds, config)
    elif task == "scan2cap":
        s = ev.calc_scan2cap_score(preds, tokenizer, scorers, config)
    elif task == "scanqa":
        s = ev.calc_scanqa_score(preds, tokenizer, scorers, config)
    else:
        s = ev.calc_sqa3d_score(preds, tokenizer, scorers, config)
        s = {k.replace("[sqa3d]", f"[{task}]"): v for k, v in s.items()}
    # fractions -> percent, as papers report them
    return {k: 100 * float(v) for k, v in s.items()}


def score_real3dqa(pred_dir, out, cfg, name):
    import evaluate_debiased_exact_match as dem
    import evaluate_rotation_robustness as rot
    res, files = {"empty_preds": {}}, {}
    for split in REAL3DQA_SPLITS:
        task = f"real3dqa_{split}"
        preds = load_preds(pred_dir, task, cfg[task])
        if preds is None:
            return None
        rows = []
        for p in preds:
            empty = not dem.clean_answer(p["pred"]).strip()
            res["empty_preds"][split] = res["empty_preds"].get(split, 0) + empty
            # int ids: the rotation script int()s the suffix-stripped 90/180/270 ids
            # but not the 0-degree ones, so string ids would never intersect
            rows.append({"question_id": int(p["qid"]), "scene_id": p["scene_id"],
                         "response_gt": p["ref_captions"], "response_pred": EMPTY if empty else p["pred"],
                         "prompt": p["prompt"]})
        files[split] = out / f"pred_{split}.json"
        json.dump(rows, open(files[split], "w"), indent=1)
        res[split] = dem.evaluate_em(str(files[split]))
    res["rotation"] = rot.quick_consistency_score([str(files[s]) for s in ROTATIONS])
    # the same numbers through the unmodified scripts' CLIs, for the record
    log = []
    for cmd in [[sys.executable, str(HERE / "real3dqa" / "evaluate_debiased_exact_match.py"),
                 str(files["debiased_test"]), "--name", name],
                [sys.executable, str(HERE / "real3dqa" / "evaluate_rotation_robustness.py")]
                + [str(files[s]) for s in ROTATIONS] + ["--name", name]]:
        log.append(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    (out / "official_scripts.txt").write_text("\n".join(log).replace(str(REPO) + "/", ""))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred_dir", required=True, help="run_infer.sh output dir")
    ap.add_argument("--config", help="config with val_file_dict (default <pred_dir>/annos/"
                                     "extra_eval_config.py, else beacon3d_config.py)")
    ap.add_argument("--result_dir", default="extra_eval/results")
    ap.add_argument("--benches", default="scanrefer,multi3dref,scan2cap,scanqa,sqa3d,sqa3d_test,real3dqa")
    ap.add_argument("--name", default="GPT4Scene")
    args = ap.parse_args()
    pred_dir = os.path.abspath(args.pred_dir)
    config = args.config or os.path.join(pred_dir, "annos", "extra_eval_config.py")
    if not os.path.exists(config) and not args.config:  # a run without sqa3d_test / real3dqa
        config = os.path.join(pred_dir, "annos", "beacon3d_config.py")
    config = os.path.abspath(config)
    result_dir = Path(args.result_dir).resolve()
    os.chdir(REPO)  # val_file_dict paths are relative to the repo root
    cfg = {}
    exec(open(config).read(), cfg)
    cfg = {k: os.path.abspath(v) for k, v in cfg["val_file_dict"].items() if isinstance(v, str)}
    os.chdir(REPO / "evaluate")  # utils/eval.py reads annotation/... relative to evaluate/
    summary_file = result_dir / "summary.json"
    summary = json.load(open(summary_file)) if summary_file.exists() else {}
    for bench in args.benches.split(","):
        out = result_dir / bench
        if bench == "real3dqa":
            out.mkdir(parents=True, exist_ok=True)
            res = score_real3dqa(pred_dir, out, cfg, args.name)
        else:
            preds = load_preds(pred_dir, bench, cfg[bench])
            res = None if preds is None else score_upstream(bench, preds)
            if res is not None:
                out.mkdir(parents=True, exist_ok=True)
                json.dump(preds, open(out / "preds.json", "w"), indent=1)
        if res is None:
            print(f"[{bench}] no predictions in {pred_dir}, skipped")
            continue
        res = {"pred_dir": os.path.relpath(pred_dir, REPO), **res}
        json.dump(res, open(out / "scores.json", "w"), indent=1)
        summary[bench] = res
        print(f"[{bench}]", json.dumps(res, indent=1), flush=True)
    json.dump(summary, open(summary_file, "w"), indent=1)


if __name__ == "__main__":
    main()
