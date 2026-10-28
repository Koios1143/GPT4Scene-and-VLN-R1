#!/usr/bin/env python3
"""Item-level agreement of two run_infer.sh output dirs (e.g. upstream vs PREFIX_CACHE=1).

For every task present in both: exact-match rate of the generated text, and for grounding
tasks (``<OBJxxx>`` answers) the rate of the same first predicted id. Writes a json summary.
"""

import argparse
import glob
import json
import os
import re


def load(d, task):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, f"qwenvl_{task}", "predictions_chunk_*.json"))):
        for p in json.load(open(f)):
            out.setdefault(p["qid"], p["pred"])
    return out


def first_id(text):
    m = re.search(r"<OBJ(\d{3})>", text)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--a", required=True, help="reference output dir (upstream path)")
    ap.add_argument("--b", required=True, help="output dir to compare (prefix-cache path)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tasks = sorted({os.path.basename(p)[len("qwenvl_"):] for p in glob.glob(os.path.join(args.a, "qwenvl_*"))}
                   & {os.path.basename(p)[len("qwenvl_"):] for p in glob.glob(os.path.join(args.b, "qwenvl_*"))})
    res = {}
    for task in tasks:
        a, b = load(args.a, task), load(args.b, task)
        common = sorted(set(a) & set(b), key=str)
        if not common:
            continue
        r = {"items": len(common), "same_text": round(100 * sum(a[q] == b[q] for q in common) / len(common), 2)}
        if any(first_id(a[q]) for q in common):
            r["same_first_obj_id"] = round(100 * sum(first_id(a[q]) == first_id(b[q]) for q in common) / len(common), 2)
        res[task] = r
        print(task, r)
    json.dump({"a": os.path.relpath(args.a), "b": os.path.relpath(args.b), "tasks": res}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
