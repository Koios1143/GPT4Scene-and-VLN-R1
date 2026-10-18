#!/bin/bash
# GPT4Scene inference (evaluate/infer.py, one process per visible GPU, chunked as in
# evaluate/infer.sh) on Scan2Cap and Beacon3D-ScanNet.
#   bash extra_eval/run_infer.sh
#   TASKS="scan2cap" bash extra_eval/run_infer.sh        # a subset
# Paths come from env.sh (see env.sh.example); relative ones are under the repo root:
#   GPT4SCENE_ENV   venv from build_env.sh                       (default .venv)
#   MODEL_PATH      the released GPT4Scene checkpoint             (default model_outputs/<name>)
#   GPT4SCENE_ANNO  GPT4Scene-Val-Annotation, linked as evaluate/annotation
#   GPT4SCENE_DATA  GPT4Scene-All images, linked as data/ (the image paths in
#                   selected_images_mark_3D_val_32.json are relative to the repo root)
#   OUT             output dir                                     (default eval_outputs/extra_eval)
#   TASKS           space-separated val tags                       (default: scan2cap + Beacon3D)
#   BATCH_SIZE      per-GPU batch size                             (default 4, as evaluate/infer.sh)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f env.sh ] && source env.sh
unset CONDA_PREFIX CONDA_DEFAULT_ENV
PY="${GPT4SCENE_ENV:-.venv}/bin/python"
MODEL_PATH="${MODEL_PATH:-model_outputs/GPT4Scene-qwen2vl_full_sft_mark_32_3D_img512}"
for link in "evaluate/annotation:${GPT4SCENE_ANNO:-}" "data:${GPT4SCENE_DATA:-}"; do
    dst="${link%%:*}"; src="${link#*:}"
    [ -n "$src" ] && [ ! -e "$dst" ] && ln -s "$src" "$dst"
done
OUT="${OUT:-eval_outputs/extra_eval}"
TASKS="${TASKS:-scan2cap beacon3d_scannet_grounding beacon3d_scannet_qa}"
export PYTHONPATH=.
mkdir -p "$OUT"
$PY extra_eval/beacon3d/build_annos.py --out_dir "$OUT/annos" || exit 1
gpu_list="${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}
echo "model=$MODEL_PATH out=$OUT tasks=[$TASKS] chunks=$CHUNKS"
for task in $TASKS; do
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} $PY evaluate/infer.py "$OUT/annos/beacon3d_config.py" \
            evaluate True model qwenvl batch_size "${BATCH_SIZE:-4}" num_workers 8 \
            model_path "$MODEL_PATH" scene_anno evaluate/annotation/selected_images_mark_3D_val_32.json \
            save_interval 2 val_tag "$task" output_dir "$OUT" num_chunks "$CHUNKS" chunk_idx "$IDX" &
    done
    wait
    echo "finished $task"
done
echo INFER_DONE
