#!/bin/bash
# Rule "all" for Beacon3D 3RScan / MultiScan: our Mask3D run (ScanNet200 checkpoint, the setup
# behind GPT4Scene's ScanNet marks) on the meshes of run_render_rgbd.sh, then render.
#   DOMAIN=3rscan bash extra_eval/render/run_mask3d_rgbd.sh       (after run_render_rgbd.sh)
# Paths come from env.sh (see env.sh.example):
#   MASK3D_DIR        Mask3D checkout (github.com/JonasSchult/Mask3D, commit 11bd5ff)
#   MASK3D_CKPT       scannet200_val.ckpt (Mask3D release)
#   MASK3D_PY         python with Mask3D's dependencies (MinkowskiEngine, ...)
#   SEGMENTATOR       ScanNet Segmentator binary (github.com/ScanNet/ScanNet Segmentator/)
#   SCANNET200_META   dir with Mask3D's ScanNet200 label_database.yaml + color_mean_std.yaml
#   RENDER_RGBD_DIR   as run_render_rgbd.sh                        (default eval_outputs/render_rgbd)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
REPO="$PWD"
[ -f env.sh ] && source env.sh
unset CONDA_PREFIX CONDA_DEFAULT_ENV
PY="${GPT4SCENE_ENV:-.venv}/bin/python"
DOMAIN="${DOMAIN:?set DOMAIN=3rscan|multiscan}"
OUT="$(cd "${RENDER_RGBD_DIR:-eval_outputs/render_rgbd}/$DOMAIN" && pwd)"
SCENES="${SCENES:-$($PY -c "import json; print(','.join(sorted({x['scan_id'] for x in json.load(open('extra_eval/beacon3d/Beacon3D/data/$DOMAIN/${DOMAIN}_grounding.json'))})))")}"
NAME="gpt4scene_$DOMAIN"
$PY extra_eval/render/prepare_mask3d_rgbd.py --scene_dir "$OUT/scenes" --scenes "$SCENES" \
    --segmentator "$SEGMENTATOR" --scannet200_dir "$SCANNET200_META" --out_dir "$OUT/mask3d_data"
rm -rf "$MASK3D_DIR/eval_output/instance_evaluation_${NAME}_0"
OMP_NUM_THREADS=3 $MASK3D_PY "$REPO/extra_eval/render/mask3d_infer.py" --mask3d_dir "$MASK3D_DIR" \
    --checkpoint "$MASK3D_CKPT" --data_dir "$OUT/mask3d_data" --name "$NAME"
$PY extra_eval/render/mask3d_to_proposals.py --export_dir "$MASK3D_DIR/eval_output/instance_evaluation_${NAME}_0/decoder_-1" \
    --scene_dir "$OUT/scenes" --scenes "$SCENES" --labels_tsv evaluate/annotation/scannet/scannetv2-labels.combined.tsv \
    --out_dir "$OUT/mask3d"
$PY extra_eval/render/render_marks.py --dataset rgbd --scenes "$SCENES" --rule all --mark_sets "$OUT/mask3d/mark_sets.json" \
    --frames_dir "$OUT/frames" --scan_dir "$OUT/scenes" --inst_seg_dir "$OUT/mask3d/inst_seg" \
    --seg_attrs "$OUT/mask3d/seg_attrs.pt" --out_dir "$OUT/all"
echo MASK3D_RGBD_DONE
