#!/usr/bin/env python3
"""Render GPT4Scene-style inputs (32 marked frames + marked BEV) for one mark rule.

GPT4Scene's preprocessing is unreleased (upstream issues #2, #19: BEV made in Blender, code
"incomplete"), so this re-implements it from what the released val images show:

* frames: ``extract_frames.py``'s 64 frames (``linspace(0, N-1, 64)``, exact match with the
  released file names); the model gets the even-indexed 32 + the BEV, in that order with the
  BEV first, as ``selected_images_mark_3D_val_32.json``. Released frames are the raw color
  frame + marks (no warp; mean abs diff ~6 = JPEG re-encoding).
* 2D mark: filled disk BGR (0, 0, 137), radius 30 px, white bold Hershey digits (the released
  look at 1296 x 968). Drawn for a marked proposal when >= ``--min_vis`` (default 1: closest
  per-frame mark counts to the released frames of scene0164_00) of its mesh vertices
  pass a depth test in that frame, at the image centroid of those visible vertices. (The
  released marks sit near the centroid of the GT 2D instance region; we use the proposal's own
  points so no GT is needed.)
* BEV: perspective render, straight down, of the axis-aligned ``_vh_clean_2`` mesh (ceiling-
  facing faces above 1.8 m dropped), x right / y up, cropped to content, light-gray background,
  a 40 px "BEV Map" title band, and red ids with black leader lines from each marked proposal's
  box center pushed outward, as in the released BEV images (walls splay outward more in small
  rooms there, consistent with a fixed-FOV camera framed on the scene).

3RScan / MultiScan (``--dataset rgbd``): frames from ``extract_frames_rgbd.py`` (turned upright
by ``rot90``; mark size scaled with the frame size), meshes from ``prepare_scene_rgbd.py``, GT-
object marks from ``mark_sets_rgbd.py``; the BEV also drops everything > 2.1 m above the floor.

Mark sets come from ``mark_sets.py`` / ``mark_sets_rgbd.py``. Output (``--out_dir``) mirrors the release:
``images_2D/<scene>/<frame>.jpg``, ``images_3D/<scene>/<scene>_3D_with_text.jpg`` and
``selected_images.json`` (absolute image paths; GPT4Scene's ``scene_anno`` format).
"""

import argparse
import json
import os

import cv2
import numpy as np
import torch
from plyfile import PlyData

DISK_BGR = (0, 0, 137)
DISK_R = 30
FONT = cv2.FONT_HERSHEY_SIMPLEX
BEV_LONG = 1000  # px spanned by the floor-level long side
BEV_FOV = 30.0  # deg; wall splay of the released scene0164_00 BEV (a Blender render) eyeballed
BEV_BAND = 40
BEV_BG = (240, 240, 240)
DEPTH_TOL = 0.1  # m
RGBD_MAX_HEIGHT = 2.1  # m above the floor kept in 3RScan / MultiScan BEVs


def load_scene(scan_dir, s):
    ply = PlyData.read(os.path.join(scan_dir, s, f"{s}_vh_clean_2.ply"))
    v = ply["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    rgb = np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.uint8)
    faces = np.stack(ply["face"]["vertex_indices"]).astype(np.int64)
    align = np.eye(4)
    for line in open(os.path.join(scan_dir, s, f"{s}.txt")):
        if line.startswith("axisAlignment"):
            align = np.array(line.split()[-16:], dtype=np.float64).reshape(4, 4)
    return xyz, rgb, faces, align


def to_h(p):
    return np.c_[p, np.ones(len(p))]


def draw_disk(img, center, text, k=1.0):
    """k scales the released 1296-px look to other frame sizes (GPT4Scene resizes to 512 anyway)."""
    x, y = int(round(center[0])), int(round(center[1]))
    cv2.circle(img, (x, y), int(round(DISK_R * k)), DISK_BGR, -1, cv2.LINE_AA)
    scale, thick = (1.3, 5) if len(text) < 3 else (1.0, 4)
    scale, thick = scale * k, max(1, int(round(thick * k)))
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    cv2.putText(img, text, (x - tw // 2, y + th // 2), FONT, scale, (255, 255, 255), thick, cv2.LINE_AA)


def rot_uv(c, rot, w, h):
    """Pixel (u, v) of a w x h image -> its position after np.rot90(img, rot)."""
    u, v = c
    return {0: (u, v), 1: (v, w - 1 - u), 2: (w - 1 - u, h - 1 - v), 3: (h - 1 - v, u)}[rot]


def render_frames(s, frames_dir, xyz, masks, marked, min_vis):
    meta = json.load(open(os.path.join(frames_dir, s, "meta.json")))
    k_color = np.loadtxt(os.path.join(frames_dir, s, "intrinsic_color.txt"))[:3, :3]
    k_depth = np.loadtxt(os.path.join(frames_dir, s, "intrinsic_depth.txt"))[:3, :3]
    w, h = meta["color_size"]
    rot = meta.get("rot90", 0)  # 3RScan frames are stored sideways
    k = max(w, h) / 1296
    r = DISK_R * k
    out, counts = {}, []
    for fi in meta["frames"]:
        img = np.ascontiguousarray(np.rot90(cv2.imread(os.path.join(frames_dir, s, "color", f"{fi}.jpg")), rot))
        depth = cv2.imread(os.path.join(frames_dir, s, "depth", f"{fi}.png"), -1).astype(np.float64) / 1000
        pose = np.loadtxt(os.path.join(frames_dir, s, "pose", f"{fi}.txt"))
        n = 0
        if np.all(np.isfinite(pose)):
            w2c = np.linalg.inv(pose)
            for pid in marked:
                cam = (to_h(xyz[masks[pid]]) @ w2c.T)[:, :3]
                cam = cam[cam[:, 2] > 0.1]
                if len(cam) < min_vis:
                    continue
                ud = cam @ k_depth.T
                ud = ud[:, :2] / ud[:, 2:3]
                ok = (ud[:, 0] >= 0) & (ud[:, 0] < depth.shape[1]) & (ud[:, 1] >= 0) & (ud[:, 1] < depth.shape[0])
                d = np.zeros(len(cam))
                d[ok] = depth[ud[ok, 1].astype(int), ud[ok, 0].astype(int)]
                vis = ok & (d > 0) & (np.abs(d - cam[:, 2]) < DEPTH_TOL)
                if vis.sum() < min_vis:
                    continue
                uc = cam[vis] @ k_color.T
                c = np.clip((uc[:, :2] / uc[:, 2:3]).mean(0), r, [w - 1 - r, h - 1 - r])
                draw_disk(img, rot_uv(c, rot, w, h), str(pid), k)
                n += 1
        out[fi] = img
        counts.append(n)
    return out, counts


def render_bev(xyz, rgb, faces, align, boxes, marked, rng, max_height=None):
    a = (to_h(xyz) @ align.T)[:, :3]
    tri = a[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
    zfloor = np.percentile(a[:, 2], 1)
    drop = (normal[:, 2] < -0.5) & (tri[:, :, 2].mean(1) > zfloor + 1.8)  # ceiling
    if max_height is not None:  # 3RScan / MultiScan meshes have inconsistent normals: also cut by height
        drop |= tri[:, :, 2].mean(1) > zfloor + max_height
    faces = faces[~drop]
    # perspective camera straight above the scene center; FOV on the long side, distance set so
    # the floor-level extent just fits; focal length so that extent spans BEV_LONG px
    lo, hi = a[:, :2].min(0), a[:, :2].max(0)
    center, extent = (lo + hi) / 2, max(hi - lo)
    dist = (extent / 2) / np.tan(np.radians(BEV_FOV) / 2)
    zcam = zfloor + dist
    focal = BEV_LONG * dist / extent

    def project(p):
        depth = zcam - p[:, 2]
        return np.stack([focal * (p[:, 0] - center[0]) / depth, -focal * (p[:, 1] - center[1]) / depth], 1), depth

    t = a[faces]
    keep = (zcam - t[:, :, 2]).min(1) > 0.2
    faces, t = faces[keep], t[keep]
    tp = np.stack([project(t[:, k])[0] for k in range(3)], 1)
    area_px = 0.5 * np.abs((tp[:, 1, 0] - tp[:, 0, 0]) * (tp[:, 2, 1] - tp[:, 0, 1])
                           - (tp[:, 2, 0] - tp[:, 0, 0]) * (tp[:, 1, 1] - tp[:, 0, 1]))
    reps = np.clip(np.ceil(area_px * 4), 1, 4000).astype(int)  # ~4 samples per covered pixel
    fid = np.repeat(np.arange(len(faces)), reps)
    r1, r2 = rng.random(len(fid)), rng.random(len(fid))
    flip = r1 + r2 > 1
    r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
    bary = np.stack([1 - r1 - r2, r1, r2], 1)
    p = np.einsum("nk,nkd->nd", bary, t[fid])
    col = np.einsum("nk,nkd->nd", bary, rgb[faces[fid]].astype(np.float64))
    uv, depth = project(p)
    off = np.floor(uv.min(0))
    uv = (uv - off).astype(int)
    W, H = uv.max(0) + 1
    order = np.argsort(-depth)  # nearer points overwrite farther ones
    canvas = np.full((H, W, 3), BEV_BG, np.uint8)
    canvas[uv[order, 1], uv[order, 0]] = col[order][:, ::-1].astype(np.uint8)  # RGB -> BGR
    # fill isolated unsampled pixels (speckle) from their neighbours
    hole = np.all(canvas == BEV_BG, 2).astype(np.uint8)
    filled = cv2.medianBlur(canvas, 5)
    isolated = hole & (cv2.blur(hole.astype(np.float32), (5, 5)) < 0.5)
    canvas[isolated.astype(bool)] = filled[isolated.astype(bool)]
    img = np.full((H + BEV_BAND, W, 3), BEV_BG, np.uint8)
    img[:BEV_BAND] = (200, 200, 200)
    img[BEV_BAND:] = canvas
    cv2.putText(img, "BEV Map", (0, 30), FONT, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
    # red ids pushed outward from the image center along a black leader line, first free slot wins
    fscale = max(0.7, 1.1 * max(W, H) / 1000)
    fthick = 2
    anchors = project(np.array([boxes[m][:3] for m in marked], np.float64))[0] - off + [0, BEV_BAND]
    mid = np.array([W / 2, H / 2 + BEV_BAND])
    taken = []
    for pid, (cx, cy) in zip(marked, anchors):
        text = str(pid)
        (tw, th), _ = cv2.getTextSize(text, FONT, fscale, fthick)
        d = np.array([cx, cy]) - mid
        base = np.arctan2(d[1], d[0]) if np.hypot(*d) > 1 else -np.pi / 4
        for r in [40, 60, 90, 130]:
            for da in [0, 0.4, -0.4, 0.8, -0.8, 1.2, -1.2]:
                tx, ty = cx + r * fscale * np.cos(base + da), cy + r * fscale * np.sin(base + da)
                x0 = int(np.clip(tx - tw / 2, 0, W - tw))
                y0 = int(np.clip(ty + th / 2, BEV_BAND + th, H + BEV_BAND - 1))
                box = (x0, y0 - th, x0 + tw, y0)
                if not any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in taken):
                    break
            else:
                continue
            break
        taken.append(box)
        cv2.line(img, (int(cx), int(cy)), (x0 + tw // 2, y0 - th // 2), (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, text, (x0, y0), FONT, fscale, (0, 0, 255), fthick, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--dataset", default="scannet", choices=["scannet", "rgbd"],
                    help="rgbd = 3RScan / MultiScan scenes from prepare_scene_rgbd.py (GT-object marks)")
    ap.add_argument("--rule", required=True, help="key of --mark_sets")
    ap.add_argument("--mark_sets", required=True, help="mark_sets.py output")
    ap.add_argument("--frames_dir", required=True, help="extract_frames.py --out_dir")
    ap.add_argument("--scan_dir", required=True,
                    help="ScanNet scans/ (_vh_clean_2.ply, <scene>.txt) or, for rgbd, prepare_scene_rgbd.py --out_dir")
    ap.add_argument("--inst_seg_dir", help="Mask3D proposals: Chat-Scene mask3d_inst_seg/ (ScanNet) or "
                    "mask3d_to_proposals.py inst_seg/ (rgbd, rule all)")
    ap.add_argument("--seg_attrs", help="rgbd rule all: mask3d_to_proposals.py seg_attrs.pt (proposal boxes)")
    ap.add_argument("--anno_dir", help="ScanNet: GPT4Scene-Val-Annotation (mask3d box attributes)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_vis", type=int, default=1, help="visible mesh vertices needed to draw a 2D mark")
    args = ap.parse_args()
    mark_sets = json.load(open(args.mark_sets))
    marks = mark_sets[args.rule]
    if args.dataset == "scannet":
        attrs = torch.load(os.path.join(args.anno_dir, "scannet_mask3d_val_attributes.pt"), map_location="cpu")
    sel_path = os.path.join(args.out_dir, "selected_images.json")
    selected = json.load(open(sel_path)) if os.path.exists(sel_path) else {}
    stats = {}
    for s in args.scenes.split(","):
        marked = marks[s]
        if args.dataset == "scannet":
            xyz, rgb, faces, align = load_scene(args.scan_dir, s)
            inst = torch.load(os.path.join(args.inst_seg_dir, f"{s}.pt"), map_location="cpu")
            masks = [np.asarray(it["segments"], dtype=np.int64) for it in inst[:100]]
            boxes = attrs[s]["locs"].numpy()
        elif args.inst_seg_dir:  # rgbd, Mask3D proposals (mask3d_to_proposals.py)
            sc = np.load(os.path.join(args.scan_dir, f"{s}.npz"))
            xyz, rgb, faces, align = sc["xyz"], sc["rgb"], sc["faces"], sc["align"]
            inst = torch.load(os.path.join(args.inst_seg_dir, f"{s}.pt"), map_location="cpu")
            masks = [np.asarray(it["segments"], dtype=np.int64) for it in inst[:100]]
            boxes = torch.load(args.seg_attrs, map_location="cpu")[s]["locs"].numpy()
        else:  # rgbd, GT objects: mark id -> GT objectId (mark_sets_rgbd.py); box centre in the z-up frame
            sc = np.load(os.path.join(args.scan_dir, f"{s}.npz"))
            xyz, rgb, faces, align = sc["xyz"], sc["rgb"], sc["faces"], sc["align"]
            id_map = mark_sets["id_map"][s]
            masks = {m: np.nonzero(sc["vert_obj"] == oid)[0] for m, oid in ((int(m), o) for m, o in id_map.items())}
            boxes = {}
            for m, idx in masks.items():
                p = (to_h(xyz[idx]) @ align.T)[:, :3]
                boxes[m] = (p.max(0) + p.min(0)) / 2
        frames, counts = render_frames(s, args.frames_dir, xyz, masks, marked, args.min_vis)
        d2 = os.path.join(args.out_dir, "images_2D", s)
        d3 = os.path.join(args.out_dir, "images_3D", s)
        os.makedirs(d2, exist_ok=True)
        os.makedirs(d3, exist_ok=True)
        for fi, img in frames.items():
            cv2.imwrite(os.path.join(d2, f"{fi}.jpg"), img)
        bev = render_bev(xyz, rgb, faces, align, boxes, marked, np.random.default_rng(0),
                         max_height=None if args.dataset == "scannet" else RGBD_MAX_HEIGHT)
        bev_path = os.path.join(d3, f"{s}_3D_with_text.jpg")
        cv2.imwrite(bev_path, bev)
        order = sorted(frames)[::2]
        selected[s] = [os.path.abspath(bev_path)] + [os.path.abspath(os.path.join(d2, f"{fi}.jpg")) for fi in order]
        stats[s] = {"marked": len(marked), "marks_per_frame_mean": float(np.mean(counts)),
                    "frames_with_marks": int(np.sum(np.array(counts) > 0))}
        print(s, stats[s], flush=True)
    json.dump(selected, open(sel_path, "w"), indent=1)
    stats_path = os.path.join(args.out_dir, "render_stats.json")
    old = json.load(open(stats_path)) if os.path.exists(stats_path) else {}
    json.dump(old | stats, open(stats_path, "w"), indent=1)


if __name__ == "__main__":
    main()
