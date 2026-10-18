#!/bin/bash
# Score extra_eval/run_infer.sh outputs. CPU only; Scan2Cap's METEOR / PTB tokenizer need Java.
#   * Scan2Cap: the repo's own evaluate/calculate_scores.py (BLEU-4/METEOR/ROUGE-L/CIDEr @0.25/@0.50)
#   * Beacon3D: extra_eval/beacon3d/score_beacon3d.py (official evaluators; QA needs OPENAI_API_KEY)
#   bash extra_eval/score.sh
#   OUT        run_infer.sh output dir                       (default eval_outputs/extra_eval)
#   TASKS      "scan2cap beacon3d"                           (default both)
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
TASKS="${TASKS:-scan2cap beacon3d}"
mkdir -p "$RESULT_DIR"
RESULT_DIR="$(cd "$RESULT_DIR" && pwd)"
if [[ " $TASKS " == *" scan2cap "* ]]; then
    # calculate_scores.py / utils/eval.py read annotation/... relative to evaluate/
    (cd evaluate && $PY calculate_scores.py configs/eval_configs.py evaluate True model qwenvl \
        output_dir "$OUT" num_chunks "$(ls "$OUT"/qwenvl_scan2cap/predictions_chunk_*.json | wc -l)" \
        seg_val_attr_file annotation/scannet_mask3d_val_attributes.pt calculate_score_tag scan2cap) || exit 1
    mkdir -p "$RESULT_DIR/scan2cap"
    cp "$OUT/all_evaluation_results.json" "$RESULT_DIR/scan2cap/scores.json"
    $PY -c "import glob, json, sys; json.dump(sorted((r for p in sorted(glob.glob(sys.argv[1] + '/qwenvl_scan2cap/predictions_chunk_*.json')) for r in json.load(open(p))), key=lambda r: r['qid']), open(sys.argv[2], 'w'), indent=1)" \
        "$OUT" "$RESULT_DIR/scan2cap/preds.json"
fi
if [[ " $TASKS " == *" beacon3d "* ]]; then
    $PY extra_eval/beacon3d/score_beacon3d.py --preds_dir "$OUT" --anno_dir "$OUT/annos" \
        --result_dir "$RESULT_DIR/beacon3d_scannet" --qa_model "${QA_MODEL:-gpt-4o-mini-2024-07-18}" || exit 1
fi
echo SCORE_DONE
