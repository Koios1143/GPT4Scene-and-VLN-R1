#!/usr/bin/env python3
"""3RScan ``sequence.zip`` / MultiScan scan zip -> the 64 frames GPT4Scene samples per scene.

Same sampling and output layout as ``extract_frames.py`` (ScanNet): frames
``np.linspace(0, N - 1, 64).astype(int)``; ``color/<i>.jpg``, ``depth/<i>.png`` (uint16, mm),
``pose/<i>.txt`` (camera-to-world, OpenCV camera axes: x right, y down, z forward),
``intrinsic_{color,depth}.txt`` and ``meta.json``. Frames are stored as captured;
``meta.json["rot90"]`` = how many times ``np.rot90`` (counter-clockwise) turns them upright,
from the mean direction of world up in the camera images (3RScan's Tango frames are stored
sideways; MultiScan's are upright). World up = +z for 3RScan; MultiScan poses live in a y-up
ARKit frame, so up there is ``align.json``'s transform applied to the mesh's +z.

* 3RScan: ``frame-XXXXXX.{color.jpg,depth.pgm,pose.txt}`` + ``_info.txt`` (calibrated color /
  depth intrinsics, depth shift 1000); poses are already camera-to-world in OpenCV axes.
* MultiScan: ``<scene>.mp4`` (color), ``<scene>.depth.zlib`` (one raw-deflate stream of float16
  metres, 256 x 192 per frame), ``<scene>.jsonl`` (per-frame intrinsics + column-major ARKit
  camera-to-world ``transform``, converted to OpenCV axes by flipping y and z). Depth
  intrinsics = color intrinsics scaled to the depth resolution.
"""

import argparse
import io
import json
import os
import zipfile
import zlib

import cv2
import numpy as np

N_FRAMES = 64
FLIP_YZ = np.diag([1.0, -1.0, -1.0, 1.0])


def choose_rot90(poses, world_up):
    """np.rot90 count that makes world up point up (-v) in the image."""
    up = np.mean([p[:3, :3].T @ world_up for p in poses if np.all(np.isfinite(p))], 0)
    du, dv = up[0], up[1]  # image-plane direction of world up (u right, v down)
    cands = {0: (du, dv), 1: (dv, -du), 2: (-du, -dv), 3: (-dv, du)}
    return max(cands, key=lambda k: -cands[k][1])


def write(out, fi, color_bgr, depth_mm, pose):
    cv2.imwrite(os.path.join(out, "color", f"{fi}.jpg"), color_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(out, "depth", f"{fi}.png"), depth_mm.astype(np.uint16))
    np.savetxt(os.path.join(out, "pose", f"{fi}.txt"), pose)


def extract_3rscan(scan_dir, out):
    z = zipfile.ZipFile(os.path.join(scan_dir, "sequence.zip"))
    info = {}
    for line in z.read("_info.txt").decode().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k.strip()] = v.strip()
    n = int(info["m_frames.size"])
    k_color = np.array(info["m_calibrationColorIntrinsic"].split(), float).reshape(4, 4)
    k_depth = np.array(info["m_calibrationDepthIntrinsic"].split(), float).reshape(4, 4)
    shift = float(info["m_depthShift"])
    frames = np.linspace(0, n - 1, N_FRAMES).astype(int).tolist()
    poses = {}
    for fi in frames:
        color = cv2.imdecode(np.frombuffer(z.read(f"frame-{fi:06d}.color.jpg"), np.uint8), cv2.IMREAD_COLOR)
        depth = cv2.imdecode(np.frombuffer(z.read(f"frame-{fi:06d}.depth.pgm"), np.uint8), cv2.IMREAD_UNCHANGED)
        pose = np.loadtxt(io.StringIO(z.read(f"frame-{fi:06d}.pose.txt").decode()))
        poses[fi] = pose
        write(out, fi, color, depth.astype(np.float64) * 1000.0 / shift, pose)
    size = [int(info["m_colorWidth"]), int(info["m_colorHeight"])]
    dsize = [int(info["m_depthWidth"]), int(info["m_depthHeight"])]
    return n, frames, k_color, k_depth, size, dsize, poses, np.array([0.0, 0.0, 1.0])


def extract_multiscan(zip_path, scene, out, tmp):
    z = zipfile.ZipFile(zip_path)
    cams = [json.loads(l) for l in z.read(f"{scene}/{scene}.jsonl").decode().splitlines() if l.strip()]
    meta = json.loads(z.read(f"{scene}/{scene}.json"))
    color_stream = next(s for s in meta["streams"] if s["type"] == "color_camera")
    depth_stream = next(s for s in meta["streams"] if s["type"] == "lidar_sensor")
    dh, dw = depth_stream["resolution"]  # [192, 256]
    video = z.extract(f"{scene}/{scene}.mp4", tmp)
    cap = cv2.VideoCapture(video)
    n = min(len(cams), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    frames = np.linspace(0, n - 1, N_FRAMES).astype(int).tolist()
    keep = set(frames)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # depth: stream-decompress, keep only the sampled frames
    fbytes = dh * dw * 2
    depths, buf, idx = {}, bytearray(), 0
    d = zlib.decompressobj(-15)
    with z.open(f"{scene}/{scene}.depth.zlib") as f:
        while idx < n:
            chunk = f.read(1 << 22)
            buf += d.decompress(chunk) if chunk else d.flush()
            off = 0
            while len(buf) - off >= fbytes:
                if idx in keep:
                    depths[idx] = np.frombuffer(bytes(buf[off:off + fbytes]), np.float16).reshape(dh, dw).astype(np.float64)
                off, idx = off + fbytes, idx + 1
            del buf[:off]
            if not chunk:
                break
    poses, k_color = {}, None
    for fi in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, color = cap.read()
        assert ok, (scene, fi)
        cam = cams[fi]
        c2w = np.array(cam["transform"], float).reshape(4, 4).T @ FLIP_YZ
        poses[fi] = c2w
        k = np.array(cam["intrinsics"], float).reshape(3, 3).T
        k_color = np.eye(4)
        k_color[:3, :3] = k
        dm = depths[fi]
        write(out, fi, color, np.where(np.isfinite(dm), dm * 1000.0, 0), c2w)
    os.remove(video)
    to_pose = np.array(json.loads(z.read(f"{scene}/{scene}.align.json"))["coordinate_transform"], float).reshape(4, 4).T
    up = to_pose[:3, :3] @ np.array([0.0, 0.0, 1.0])  # mesh is z-up; poses are in to_pose @ mesh
    k_depth = k_color.copy()
    k_depth[0, :3] *= dw / w
    k_depth[1, :3] *= dh / h
    assert color_stream["resolution"] in ([h, w], [w, h])
    return n, frames, k_color, k_depth, [w, h], [dw, dh], poses, up / np.linalg.norm(up)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, choices=["3rscan", "multiscan"])
    ap.add_argument("--scans_dir", required=True, help="3RScan scans/<id>/ or MultiScan scans/<scene>.zip")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--scenes", required=True)
    args = ap.parse_args()
    for s in args.scenes.split(","):
        out = os.path.join(args.out_dir, s)
        if os.path.exists(os.path.join(out, "meta.json")):
            print(f"{s}: done already")
            continue
        for sub in ["color", "depth", "pose"]:
            os.makedirs(os.path.join(out, sub), exist_ok=True)
        if args.dataset == "3rscan":
            res = extract_3rscan(os.path.join(args.scans_dir, s), out)
        else:
            res = extract_multiscan(os.path.join(args.scans_dir, f"{s}.zip"), s, out, out)
        n, frames, k_color, k_depth, size, dsize, poses, world_up = res
        np.savetxt(os.path.join(out, "intrinsic_color.txt"), k_color)
        np.savetxt(os.path.join(out, "intrinsic_depth.txt"), k_depth)
        rot = choose_rot90(list(poses.values()), world_up)
        meta = {"num_frames": n, "frames": frames, "color_size": size, "depth_size": dsize, "rot90": int(rot),
                "dataset": args.dataset}
        json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
        print(f"{s}: {n} frames -> {len(frames)} kept, rot90={rot}", flush=True)


if __name__ == "__main__":
    main()
