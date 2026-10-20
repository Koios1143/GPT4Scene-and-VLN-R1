#!/bin/bash
# Re-render GPT4Scene's inputs for every Beacon3D-ScanNet scene under three mark rules
# (see mark_sets.py): released (reconstruction of the released images' rule), gt, all.
#   bash extra_eval/render/run_render.sh
# Needs ScanNet .sens for the scenes (ScanNet's download-scannet.py --type .sens --id <scene>,
# under its Terms of Use). Paths come from env.sh (see env.sh.example):
#   SCANNET_SENS_DIR   dir with <scene>/<scene>.sens
#   SCANNET_SCANS_DIR  ScanNet scans/ (<scene>_vh_clean_2.ply, <scene>.txt)
#   MASK3D_INST_SEG    Chat-Scene's mask3d_inst_seg/ (download.sh)
#   RENDER_DIR         output                                (default eval_outputs/render)
#   SCENES             comma list (default: all 16 Beacon3D-ScanNet scenes)
#   RULES              space list (default "released gt all")
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
[ -f env.sh ] && source env.sh
unset CONDA_PREFIX CONDA_DEFAULT_ENV
PY="${GPT4SCENE_ENV:-.venv}/bin/python"
OUT="${RENDER_DIR:-eval_outputs/render}"
SCENES="${SCENES:-$($PY -c "import json; print(','.join(sorted({x['scan_id'] for x in json.load(open('extra_eval/beacon3d/Beacon3D/data/scannet/scannet_grounding.json'))})))")}"
mkdir -p "$OUT"
$PY extra_eval/render/extract_frames.py --scans_dir "$SCANNET_SENS_DIR" --out_dir "$OUT/frames" --scenes "$SCENES"
$PY extra_eval/render/mark_sets.py --anno_dir evaluate/annotation --scenes "$SCENES" --out "$OUT/mark_sets.json"
for rule in ${RULES:-released gt all}; do
    $PY extra_eval/render/render_marks.py --scenes "$SCENES" --rule "$rule" --mark_sets "$OUT/mark_sets.json" \
        --frames_dir "$OUT/frames" --scan_dir "$SCANNET_SCANS_DIR" --inst_seg_dir "$MASK3D_INST_SEG" \
        --anno_dir evaluate/annotation --out_dir "$OUT/$rule"
done
echo RENDER_DONE
