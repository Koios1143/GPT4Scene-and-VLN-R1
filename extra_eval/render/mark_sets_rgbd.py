#!/usr/bin/env python3
"""3RScan / MultiScan mark sets: rule ``gt`` = every GT object except wall / floor / ceiling.

No Mask3D proposals exist for these datasets, so the marks are the GT instances themselves
(the condition of a GT-mask model such as Chat-Scene on these domains). Mark ids are
0, 1, ... over the marked objects in objectId order, like GPT4Scene's ``<OBJxxx>`` ids;
``id_map[scene][mark id] = objectId`` (Beacon3D ``target_id``) is used to score grounding
exactly (no IoU matching). Objects with no mesh vertex are skipped.
"""

import argparse
import json
import os

import numpy as np

STRUCTURE = {"wall", "floor", "ceiling"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene_dir", required=True, help="prepare_scene_rgbd.py --out_dir")
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = {"gt": {}, "id_map": {}}
    for s in args.scenes.split(","):
        sc = np.load(os.path.join(args.scene_dir, f"{s}.npz"))
        objects = {int(k): v for k, v in json.loads(str(sc["objects"])).items()}
        present = set(np.unique(sc["vert_obj"]).tolist())
        keep = [o for o in sorted(objects) if objects[o] not in STRUCTURE and o in present]
        res["gt"][s] = list(range(len(keep)))
        res["id_map"][s] = {str(m): o for m, o in enumerate(keep)}
        print(s, f"{len(keep)} marked of {len(objects)} objects")
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
