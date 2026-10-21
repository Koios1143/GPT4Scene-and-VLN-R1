#!/usr/bin/env python3
"""3RScan / MultiScan scan -> one ``<out_dir>/<scene>.npz`` the renderer reads.

Keys: ``xyz`` (mesh vertices, in the frame of the camera poses of ``extract_frames_rgbd.py``),
``rgb`` (uint8 vertex colors), ``faces``, ``align`` (4x4, pose frame -> z-up frame used for the
BEV), ``vert_obj`` (GT objectId per vertex, 0 = none) and ``objects`` (json: objectId -> label).
GT objectIds are the ids Beacon3D's ``target_id`` uses for these domains.

* 3RScan: ``labels.instances.annotated.v2.ply`` (vertices, faces, per-vertex objectId) +
  colors sampled from ``mesh.refined_0.png`` at the per-vertex UVs of ``mesh.refined.v2.obj``
  (same vertices, ``f v/vt`` with v == vt); labels from ``semseg.v2.json``; already z-up.
* MultiScan: ``<scene>.ply`` (vertex colors, per-face objectId -> per-vertex by majority of
  incident faces) is z-up; the ARKit camera poses live in ``T @ mesh`` with ``T`` =
  ``align.json`` ``coordinate_transform`` (column-major) -- depth back-projected with the
  poses lands within ~1 cm of ``T @ mesh`` vs 0.3-1.3 m otherwise. So ``xyz = T @ mesh`` and
  ``align = T^-1``. Labels from ``annotations.json`` (``"<label>.<k>"`` -> ``<label>``).
"""

import argparse
import json
import os
import zipfile

import cv2
import numpy as np
from plyfile import PlyData


def prep_3rscan(d):
    v = PlyData.read(os.path.join(d, "labels.instances.annotated.v2.ply"))
    vx = v["vertex"]
    xyz = np.stack([vx["x"], vx["y"], vx["z"]], 1).astype(np.float64)
    faces = np.stack(v["face"]["vertex_indices"]).astype(np.int64)
    vert_obj = np.asarray(vx["objectId"], np.int64)
    uv = np.array([l.split()[1:3] for l in open(os.path.join(d, "mesh.refined.v2.obj")) if l.startswith("vt ")], float)
    assert len(uv) == len(xyz)
    tex = cv2.imread(os.path.join(d, "mesh.refined_0.png"))[:, :, ::-1]
    th, tw = tex.shape[:2]
    px = np.clip((uv[:, 0] * (tw - 1)).round().astype(int), 0, tw - 1)
    py = np.clip(((1 - uv[:, 1]) * (th - 1)).round().astype(int), 0, th - 1)
    rgb = tex[py, px]
    objects = {g["objectId"]: g["label"] for g in json.load(open(os.path.join(d, "semseg.v2.json")))["segGroups"]}
    return xyz, rgb, faces, np.eye(4), vert_obj, objects


def prep_multiscan(zip_path, scene, tmp):
    z = zipfile.ZipFile(zip_path)
    ply = PlyData.read(z.extract(f"{scene}/{scene}.ply", tmp))
    vx, fc = ply["vertex"], ply["face"]
    xyz = np.stack([vx["x"], vx["y"], vx["z"]], 1).astype(np.float64)
    rgb = np.stack([vx["red"], vx["green"], vx["blue"]], 1).astype(np.uint8)
    faces = np.stack(fc["vertex_indices"]).astype(np.int64)
    face_obj = np.asarray(fc["objectId"], np.int64)
    # per-vertex objectId = most frequent objectId among incident faces
    vi = faces.reshape(-1)
    fo = np.repeat(face_obj, faces.shape[1])
    pairs, counts = np.unique(np.stack([vi, fo], 1), axis=0, return_counts=True)
    order = np.lexsort((-counts, pairs[:, 0]))
    pairs = pairs[order]
    first = np.r_[True, pairs[1:, 0] != pairs[:-1, 0]]
    vert_obj = np.zeros(len(xyz), np.int64)
    vert_obj[pairs[first, 0]] = pairs[first, 1]
    to_pose = np.array(json.loads(z.read(f"{scene}/{scene}.align.json"))["coordinate_transform"], float).reshape(4, 4).T
    xyz = (np.c_[xyz, np.ones(len(xyz))] @ to_pose.T)[:, :3]
    align = np.linalg.inv(to_pose)
    ann = json.loads(z.read(f"{scene}/{scene}.annotations.json"))
    objects = {o["objectId"]: o["label"].rsplit(".", 1)[0] for o in ann["objects"]}
    os.remove(os.path.join(tmp, scene, f"{scene}.ply"))
    return xyz, rgb, faces, align, vert_obj, objects


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, choices=["3rscan", "multiscan"])
    ap.add_argument("--scans_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--scenes", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for s in args.scenes.split(","):
        out = os.path.join(args.out_dir, f"{s}.npz")
        if os.path.exists(out):
            print(f"{s}: done already")
            continue
        if args.dataset == "3rscan":
            xyz, rgb, faces, align, vert_obj, objects = prep_3rscan(os.path.join(args.scans_dir, s))
        else:
            xyz, rgb, faces, align, vert_obj, objects = prep_multiscan(os.path.join(args.scans_dir, f"{s}.zip"), s, args.out_dir)
        np.savez_compressed(out, xyz=xyz, rgb=rgb, faces=faces, align=align, vert_obj=vert_obj,
                            objects=json.dumps({int(k): v for k, v in objects.items()}))
        print(f"{s}: {len(xyz)} verts, {len(faces)} faces, {len(objects)} objects, "
              f"{(vert_obj > 0).mean():.2f} of verts labelled", flush=True)


if __name__ == "__main__":
    main()
