#!/bin/bash
# Render GPT4Scene's inputs for Beacon3D's 3RScan / MultiScan scenes (rule gt: GT-object marks).
#   DOMAIN=3rscan bash extra_eval/render/run_render_rgbd.sh
#   DOMAIN=multiscan bash extra_eval/render/run_render_rgbd.sh
# Paths come from env.sh (see env.sh.example):
#   RSCAN_RAW_DIR      3RScan scans/<id>/ (official download script: sequence.zip, mesh.refined.v2.obj,
#                      mesh.refined_0.png, labels.instances.annotated.v2.ply, semseg.v2.json)
#   MULTISCAN_RAW_DIR  MultiScan scans/<scene>.zip (HF 3dlg-hcvc/MultiScan, gated)
#   RENDER_RGBD_DIR    output root; <DOMAIN>/ under it        (default eval_outputs/render_rgbd)
#   SCENES             comma list (default: all Beacon3D scenes of the domain)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
[ -f env.sh ] && source env.sh
unset CONDA_PREFIX CONDA_DEFAULT_ENV
PY="${GPT4SCENE_ENV:-.venv}/bin/python"
DOMAIN="${DOMAIN:?set DOMAIN=3rscan|multiscan}"
case "$DOMAIN" in
    3rscan) RAW="$RSCAN_RAW_DIR" ;;
    multiscan) RAW="$MULTISCAN_RAW_DIR" ;;
    *) echo "bad DOMAIN $DOMAIN"; exit 1 ;;
esac
OUT="${RENDER_RGBD_DIR:-eval_outputs/render_rgbd}/$DOMAIN"
SCENES="${SCENES:-$($PY -c "import json; print(','.join(sorted({x['scan_id'] for x in json.load(open('extra_eval/beacon3d/Beacon3D/data/$DOMAIN/${DOMAIN}_grounding.json'))})))")}"
mkdir -p "$OUT"
$PY extra_eval/render/extract_frames_rgbd.py --dataset "$DOMAIN" --scans_dir "$RAW" --out_dir "$OUT/frames" --scenes "$SCENES"
$PY extra_eval/render/prepare_scene_rgbd.py --dataset "$DOMAIN" --scans_dir "$RAW" --out_dir "$OUT/scenes" --scenes "$SCENES"
$PY extra_eval/render/mark_sets_rgbd.py --scene_dir "$OUT/scenes" --scenes "$SCENES" --out "$OUT/mark_sets.json"
$PY extra_eval/render/render_marks.py --dataset rgbd --scenes "$SCENES" --rule gt --mark_sets "$OUT/mark_sets.json" \
    --frames_dir "$OUT/frames" --scan_dir "$OUT/scenes" --out_dir "$OUT/gt"
echo RENDER_RGBD_DONE
