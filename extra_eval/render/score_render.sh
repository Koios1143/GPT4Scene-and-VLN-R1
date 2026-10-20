#!/bin/bash
# Score the Beacon3D-ScanNet runs on re-rendered inputs (one per mark rule) with the official
# evaluators, and compare the "released" rule against the released images on the validation
# scenes. CPU only; QA needs OPENAI_API_KEY.
#   bash extra_eval/render/score_render.sh
#   RULES        space list                        (default "released gt all")
#   RUN_PREFIX   inference output dirs = <prefix><rule> (default eval_outputs/render_)
#   RENDER_DIR   run_render.sh output              (default eval_outputs/render)
#   VAL_SCENES   validation scenes                 (default scene0025_00,scene0164_00,scene0426_00)
#   QA_MODEL     Beacon3D QA judge                 (default gpt-4o-mini-2024-07-18)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
[ -f env.sh ] && source env.sh
unset CONDA_PREFIX CONDA_DEFAULT_ENV
PY="${GPT4SCENE_ENV:-.venv}/bin/python"
RES=extra_eval/results
RENDER_DIR="${RENDER_DIR:-eval_outputs/render}"
mkdir -p "$RES/render"
cp "$RENDER_DIR/mark_sets.json" "$RES/render/mark_sets.json"
for rule in ${RULES:-released gt all}; do
    run="${RUN_PREFIX:-eval_outputs/render_}$rule"
    $PY extra_eval/beacon3d/score_beacon3d.py --preds_dir "$run" --anno_dir "$run/annos" \
        --result_dir "$RES/beacon3d_scannet_render_$rule" --qa_model "${QA_MODEL:-gpt-4o-mini-2024-07-18}" || exit 1
    cp "$RENDER_DIR/$rule/render_stats.json" "$RES/render/render_stats_$rule.json"
done
$PY extra_eval/render/compare_validation.py --released "$RES/beacon3d_scannet" \
    --rendered "$RES/beacon3d_scannet_render_released" \
    --scenes "${VAL_SCENES:-scene0025_00,scene0164_00,scene0426_00}" --out "$RES/render/validation.json"
echo SCORE_RENDER_DONE
