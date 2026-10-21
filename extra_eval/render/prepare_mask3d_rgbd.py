#!/usr/bin/env python3
"""3RScan / MultiScan meshes (``prepare_scene_rgbd.py``) -> a Mask3D scannet200-layout dataset.

Per scene ``validation/<scene>.npy`` with Mask3D's 12 columns (``scannet_preprocessing.py``):
xyz (the z-up frame, ``align @ xyz``), rgb 0-255, vertex normals (area-weighted face normals),
segment id (ScanNet Segmentator on the mesh, kThresh 0.01 / segMinVerts 20 as for ScanNet's
``_vh_clean_2.0.010000.segs.json``), semantic label 0 and instance id = GT objectId. Labels
only let Mask3D's validation loop run (semantic 0 is filtered out, so its AP printout is
meaningless); predictions do not use them. ``validation_database.yaml`` / ``train_database.yaml``
+ ``instance_gt/validation/<scene>.txt`` as Mask3D expects; ``label_database.yaml`` /
``color_mean_std.yaml`` (ScanNet200 train statistics) are copied from ``--scannet200_dir``.
"""

import argparse
import json
import os
import shutil
import subprocess

import numpy as np
import yaml
from plyfile import PlyData, PlyElement


def vertex_normals(xyz, faces):
    t = xyz[faces]
    fn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])  # length = 2 x area
    n = np.zeros_like(xyz)
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def segment(segmentator, xyz, rgb, faces, tmp_ply):
    v = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v["x"], v["y"], v["z"] = xyz.T
    v["red"], v["green"], v["blue"] = rgb.T
    f = np.empty(len(faces), dtype=[("vertex_indices", "i4", (3,))])
    f["vertex_indices"] = faces
    PlyData([PlyElement.describe(v, "vertex"), PlyElement.describe(f, "face")], text=False).write(tmp_ply)
    subprocess.run([segmentator, tmp_ply, "0.01", "20"], check=True, capture_output=True)
    out = tmp_ply[:-4] + ".0.010000.segs.json"
    seg = np.array(json.load(open(out))["segIndices"])
    os.remove(tmp_ply)
    os.remove(out)
    return seg


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene_dir", required=True, help="prepare_scene_rgbd.py --out_dir")
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--segmentator", required=True, help="ScanNet Segmentator binary")
    ap.add_argument("--scannet200_dir", required=True, help="dir with ScanNet200 label_database.yaml + color_mean_std.yaml")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    out = os.path.abspath(args.out_dir)
    os.makedirs(os.path.join(out, "validation"), exist_ok=True)
    os.makedirs(os.path.join(out, "instance_gt", "validation"), exist_ok=True)
    for f in ["label_database.yaml", "color_mean_std.yaml"]:
        shutil.copy(os.path.join(args.scannet200_dir, f), os.path.join(out, f))
    db = []
    for s in args.scenes.split(","):
        sc = np.load(os.path.join(args.scene_dir, f"{s}.npz"))
        xyz = (np.c_[sc["xyz"], np.ones(len(sc["xyz"]))] @ sc["align"].T)[:, :3]
        faces = sc["faces"]
        seg = segment(args.segmentator, xyz, sc["rgb"], faces, os.path.join(out, f"{s}.ply"))
        seg_ids = np.unique(seg, return_inverse=True)[1]
        inst = sc["vert_obj"].astype(np.float64)
        pts = np.hstack([xyz, sc["rgb"].astype(np.float64), vertex_normals(xyz, faces), seg_ids[:, None],
                         np.zeros((len(xyz), 1)), inst[:, None]]).astype(np.float32)
        npy = os.path.join(out, "validation", f"{s}.npy")
        np.save(npy, pts)
        gt_txt = os.path.join(out, "instance_gt", "validation", f"{s}.txt")
        np.savetxt(gt_txt, (inst + 1).astype(np.int64), fmt="%d")
        # Mask3D names a scene's export after raw_filepath's parent dir (never opened)
        db.append({"filepath": npy, "scene": s, "raw_filepath": os.path.join(out, "raw", s, f"{s}.ply"), "file_len": len(xyz),
                   "instance_gt_filepath": gt_txt, "num_superpoints": int(seg_ids.max() + 1)})
        print(s, len(xyz), "verts", int(seg_ids.max() + 1), "segments", flush=True)
    for split in ["validation", "train"]:
        yaml.safe_dump(db, open(os.path.join(out, f"{split}_database.yaml"), "w"))


if __name__ == "__main__":
    main()
