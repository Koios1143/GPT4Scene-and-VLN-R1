#!/usr/bin/env python3
"""ScanNet ``<scene>.sens`` -> the 64 frames GPT4Scene samples per scene.

GPT4Scene's released frames (``images_2D/val/2D_image_mark_val_64imgs/<scene>/<i>.jpg``)
are frames ``np.linspace(0, N - 1, 64).astype(int)`` of the N-frame scan (checked against
the released file names of scene0011_00 / scene0025_00 / scene0164_00), at the color
resolution (1296 x 968). For each of them this writes, under ``<out_dir>/<scene>/``:
``color/<i>.jpg`` (the .sens JPEG bytes, not re-encoded), ``depth/<i>.png`` (uint16, mm),
``pose/<i>.txt`` (camera-to-world 4x4) and once ``intrinsic_{color,depth}.txt`` +
``meta.json`` (frame count, sizes, depth shift).

Minimal reader for the .sens v4 layout of ScanNet's ``SensorData.py``; streams the file so
only the selected frames are decoded.
"""

import argparse
import json
import os
import struct
import zlib

import cv2
import numpy as np

N_FRAMES = 64


def read_mat(f):
    return np.asarray(struct.unpack("f" * 16, f.read(64)), dtype=np.float32).reshape(4, 4)


def extract(sens, out):
    with open(sens, "rb") as f:
        version = struct.unpack("I", f.read(4))[0]
        assert version == 4, version
        name_len = struct.unpack("Q", f.read(8))[0]
        f.read(name_len)
        k_color, _, k_depth, _ = read_mat(f), read_mat(f), read_mat(f), read_mat(f)
        color_comp, depth_comp = struct.unpack("ii", f.read(8))
        assert color_comp == 2 and depth_comp == 1, (color_comp, depth_comp)  # jpeg, zlib_ushort
        cw, ch, dw, dh = struct.unpack("IIII", f.read(16))
        depth_shift = struct.unpack("f", f.read(4))[0]
        n = struct.unpack("Q", f.read(8))[0]
        keep = set(np.linspace(0, n - 1, N_FRAMES).astype(int).tolist())
        for sub in ["color", "depth", "pose"]:
            os.makedirs(os.path.join(out, sub), exist_ok=True)
        for i in range(n):
            pose = read_mat(f)
            f.read(16)  # timestamps
            csize, dsize = struct.unpack("QQ", f.read(16))
            if i not in keep:
                f.seek(csize + dsize, 1)
                continue
            color, depth = f.read(csize), f.read(dsize)
            with open(os.path.join(out, "color", f"{i}.jpg"), "wb") as g:
                g.write(color)
            d = np.frombuffer(zlib.decompress(depth), dtype=np.uint16).reshape(dh, dw)
            cv2.imwrite(os.path.join(out, "depth", f"{i}.png"), d)
            np.savetxt(os.path.join(out, "pose", f"{i}.txt"), pose)
    np.savetxt(os.path.join(out, "intrinsic_color.txt"), k_color)
    np.savetxt(os.path.join(out, "intrinsic_depth.txt"), k_depth)
    meta = {"num_frames": n, "frames": sorted(keep), "color_size": [cw, ch], "depth_size": [dw, dh],
            "depth_shift": depth_shift}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scans_dir", required=True, help="dir with <scene>/<scene>.sens")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--scenes", required=True, help="comma-separated scene ids")
    args = ap.parse_args()
    for s in args.scenes.split(","):
        out = os.path.join(args.out_dir, s)
        if os.path.exists(os.path.join(out, "meta.json")):
            print(f"{s}: done already")
            continue
        meta = extract(os.path.join(args.scans_dir, s, f"{s}.sens"), out)
        print(f"{s}: {meta['num_frames']} frames -> {len(meta['frames'])} kept")


if __name__ == "__main__":
    main()
