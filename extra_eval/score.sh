#!/bin/bash
# Score extra_eval/run_infer.sh outputs. CPU only; pycocoevalcap (Scan2Cap/ScanQA/SQA3D) needs Java.
#   * ScanRefer, Multi3DRefer, Scan2Cap, ScanQA, SQA3D val/test, Real-3DQA: extra_eval/score_extra.py
#     (the repo's own evaluate/utils/eval.py scorers; official Real-3DQA evaluators)
#   * Beacon3D: extra_eval/beacon3d/score_beacon3d.py (official evaluators; QA needs OPENAI_API_KEY)
#   bash extra_eval/score.sh
#   OUT        run_infer.sh output dir                       (default eval_outputs/extra_eval)
#   TASKS      any of "scanrefer multi3dref scan2cap scanqa sqa3d sqa3d_test real3dqa beacon3d"
#              (default all)
#   QA_MODEL   Beacon3D QA judge (default gpt-4o-mini-2024-07-18; upstream gpt-4o-2024-11-20)
#   RESULT_DIR where results go                              (default extra_eval/results)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$PWD"
[ -f env.sh ] && source env.sh
unset CONDA_PREFIX CONDA_DEFAULT_ENV
PY="$(cd "$(dirname "${GPT4SCENE_ENV:-.venv}/bin/python")" && pwd)/python"
OUT="$(cd "${OUT:-eval_outputs/extra_eval}" && pwd)"
RESULT_DIR="${RESULT_DIR:-extra_eval/results}"
TASKS="${TASKS:-scanrefer multi3dref scan2cap scanqa sqa3d sqa3d_test real3dqa beacon3d}"
mkdir -p "$RESULT_DIR"
RESULT_DIR="$(cd "$RESULT_DIR" && pwd)"
BENCHES="$(echo " $TASKS " | sed 's/ beacon3d / /' | xargs | tr ' ' ',')"
if [ -n "$BENCHES" ]; then
    $PY extra_eval/score_extra.py --pred_dir "$OUT" --result_dir "$RESULT_DIR" --benches "$BENCHES" || exit 1
fi
if [[ " $TASKS " == *" beacon3d "* ]]; then
    $PY extra_eval/beacon3d/score_beacon3d.py --preds_dir "$OUT" --anno_dir "$OUT/annos" \
        --result_dir "$RESULT_DIR/beacon3d_scannet" --qa_model "${QA_MODEL:-gpt-4o-mini-2024-07-18}" || exit 1
fi
echo SCORE_DONE
