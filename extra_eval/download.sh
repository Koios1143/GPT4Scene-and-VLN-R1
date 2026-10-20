#!/bin/bash
# Released GPT4Scene checkpoint, val annotations and val images (the repo's download.py
# minus the training-only parts), into the dirs named in env.sh. Idempotent.
#   bash extra_eval/download.sh
#   MODEL_PATH       alexzyqi/GPT4Scene-qwen2vl_full_sft_mark_32_3D_img512   (default model_outputs/<name>)
#   GPT4SCENE_ANNO   alexzyqi/GPT4Scene-Val-Annotation                       (default evaluate/annotation)
#   GPT4SCENE_DATA   alexzyqi/GPT4Scene-All, images_{2D,3D}.zip unzipped    (default data)
#   MASK3D_INST_SEG  ZzZZCHS/Chat-Scene data/mask3d_inst_seg.tar.gz, unpacked (default data/mask3d_inst_seg);
#                    the Mask3D masks behind GPT4Scene's <OBJxxx> marks, for extra_eval/render/
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f env.sh ] && source env.sh
PY="${GPT4SCENE_ENV:-.venv}/bin/python"
MODEL_PATH="${MODEL_PATH:-model_outputs/GPT4Scene-qwen2vl_full_sft_mark_32_3D_img512}"
ANNO="${GPT4SCENE_ANNO:-evaluate/annotation}"
DATA="${GPT4SCENE_DATA:-data}"
INST="${MASK3D_INST_SEG:-data/mask3d_inst_seg}"
$PY - "$MODEL_PATH" "$ANNO" "$DATA" "$(dirname "$INST")" <<'PYEOF'
import sys
from huggingface_hub import hf_hub_download, snapshot_download as get
model, anno, data, inst_parent = sys.argv[1:]
get("alexzyqi/GPT4Scene-Val-Annotation", repo_type="dataset", local_dir=anno)
get("alexzyqi/GPT4Scene-All", repo_type="dataset", local_dir=data,
    allow_patterns=["images_2D.zip", "images_3D.zip", "README.md", "dataset_info.json"])
get("alexzyqi/GPT4Scene-qwen2vl_full_sft_mark_32_3D_img512", local_dir=model)
hf_hub_download("ZzZZCHS/Chat-Scene", "data/mask3d_inst_seg.tar.gz", repo_type="dataset", local_dir=inst_parent)
PYEOF
if [ ! -d "$INST" ]; then
    tar -xzf "$(dirname "$INST")/data/mask3d_inst_seg.tar.gz" -C "$(dirname "$INST")"
    [ "$(dirname "$INST")/mask3d_inst_seg" = "$INST" ] || mv "$(dirname "$INST")/mask3d_inst_seg" "$INST"
fi
for z in images_2D images_3D; do
    if [ -f "$DATA/$z.zip" ] && [ ! -d "$DATA/$z" ]; then
        (cd "$DATA" && unzip -q "$z.zip")
    fi
done
ls "$DATA"
echo DOWNLOAD_DONE
