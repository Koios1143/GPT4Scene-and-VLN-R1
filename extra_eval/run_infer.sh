#!/bin/bash
# GPT4Scene inference on ScanRefer, Multi3DRefer, Scan2Cap, ScanQA, SQA3D (val + test),
# Real-3DQA and Beacon3D, one process per visible GPU.
#   bash extra_eval/run_infer.sh                                   # upstream path (evaluate/infer.py)
#   PREFIX_CACHE=1 bash extra_eval/run_infer.sh                    # scene-prefix KV cache, same outputs
#   TASKS="scanqa sqa3d" PREFIX_CACHE=1 bash extra_eval/run_infer.sh   # a subset
# Paths come from env.sh (see env.sh.example); relative ones are under the repo root:
#   GPT4SCENE_ENV   venv from build_env.sh                       (default .venv)
#   MODEL_PATH      the released GPT4Scene checkpoint             (default model_outputs/<name>)
#   GPT4SCENE_ANNO  GPT4Scene-Val-Annotation, linked as evaluate/annotation
#   GPT4SCENE_DATA  GPT4Scene-All images, linked as data/ (the image paths in
#                   selected_images_mark_3D_val_32.json are relative to the repo root)
#   SQA3D_TEST      Chat-Scene's annotations/sqa3d_test.json      (for sqa3d_test)
#   REAL3DQA_DIR    Real-3DQA <split>.jsonl (HF Oliver-Ma/Real-3DQA)   (for real3dqa_*)
#   SCENE_ANNO      per-scene image list (default: the released
#                   evaluate/annotation/selected_images_mark_3D_val_32.json; or a
#                   render/run_render.sh <rule>/selected_images.json)
#   DOMAIN          Beacon3D domain of the beacon3d_* tags         (default scannet; 3rscan /
#                   multiscan need a render/run_render_rgbd.sh SCENE_ANNO)
#   OUT             output dir                                     (default eval_outputs/extra_eval)
#   TASKS           space-separated val tags                       (default: all of the above on
#                   the released images: scanrefer multi3dref scan2cap scanqa sqa3d sqa3d_test
#                   real3dqa_{debiased_test,rotation_0,rotation_90,rotation_180,rotation_270}
#                   beacon3d_scannet_{grounding,qa})
#   BATCH_SIZE      upstream path: per-GPU batch size              (default 4, as evaluate/infer.sh)
#   PREFIX_CACHE    1: extra_eval/infer_prefix_cache.py, all tasks in one pass, each scene's
#                   image prefix encoded once                      (default 0)
#   GEN_BATCH       with PREFIX_CACHE=1: questions per batched greedy decode (default 32)
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
REAL3DQA="real3dqa_debiased_test real3dqa_rotation_0 real3dqa_rotation_90 real3dqa_rotation_180 real3dqa_rotation_270"
TASKS="${TASKS:-scanrefer multi3dref scan2cap scanqa sqa3d sqa3d_test $REAL3DQA beacon3d_scannet_grounding beacon3d_scannet_qa}"
export PYTHONPATH=.
mkdir -p "$OUT"
SCENE_ANNO="${SCENE_ANNO:-evaluate/annotation/selected_images_mark_3D_val_32.json}"
$PY extra_eval/beacon3d/build_annos.py --domain "${DOMAIN:-scannet}" --scene_anno "$SCENE_ANNO" --out_dir "$OUT/annos" || exit 1
CONFIG="$OUT/annos/beacon3d_config.py"
if [[ " $TASKS " == *" sqa3d_test "* || " $TASKS " == *" real3dqa_"* ]]; then
    $PY extra_eval/build_annos.py --sqa3d_test "${SQA3D_TEST:?set SQA3D_TEST}" \
        --real3dqa_dir "${REAL3DQA_DIR:?set REAL3DQA_DIR}" --scene_anno "$SCENE_ANNO" \
        --base_config "$CONFIG" --out_dir "$OUT/annos" --config_out "$OUT/annos/extra_eval_config.py" || exit 1
    CONFIG="$OUT/annos/extra_eval_config.py"
fi
gpu_list="${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}
echo "model=$MODEL_PATH scene_anno=$SCENE_ANNO out=$OUT tasks=[$TASKS] chunks=$CHUNKS prefix_cache=${PREFIX_CACHE:-0}"
if [ "${PREFIX_CACHE:-0}" = 1 ]; then
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} $PY extra_eval/infer_prefix_cache.py --config "$CONFIG" \
            --tasks "$TASKS" --model_path "$MODEL_PATH" --scene_anno "$SCENE_ANNO" --output_dir "$OUT" \
            --num_chunks "$CHUNKS" --chunk_idx "$IDX" --gen_batch "${GEN_BATCH:-32}" &
    done
    wait
else
    for task in $TASKS; do
        for IDX in $(seq 0 $((CHUNKS-1))); do
            CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} $PY evaluate/infer.py "$CONFIG" \
                evaluate True model qwenvl batch_size "${BATCH_SIZE:-4}" num_workers 8 \
                model_path "$MODEL_PATH" scene_anno "$SCENE_ANNO" \
                save_interval 2 val_tag "$task" output_dir "$OUT" num_chunks "$CHUNKS" chunk_idx "$IDX" &
        done
        wait
        echo "finished $task"
    done
fi
echo INFER_DONE
