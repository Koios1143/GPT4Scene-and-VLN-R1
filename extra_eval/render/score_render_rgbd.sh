#!/bin/bash
# Score the Beacon3D 3RScan / MultiScan runs on re-rendered inputs (rules gt and all) with the
# official evaluators. CPU only; QA needs OPENAI_API_KEY.
#   bash extra_eval/render/score_render_rgbd.sh
#   DOMAINS          space list                            (default "3rscan multiscan")
#   RULES            space list                            (default "gt all")
#   RUN_PREFIX       inference output dirs = <prefix><domain>_<rule> (default eval_outputs/render_)
#   RENDER_RGBD_DIR  run_render_rgbd.sh output root        (default eval_outputs/render_rgbd)
#   QA_MODEL         Beacon3D QA judge                     (default gpt-4o-mini-2024-07-18)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
[ -f env.sh ] && source env.sh
unset CONDA_PREFIX CONDA_DEFAULT_ENV
PY="${GPT4SCENE_ENV:-.venv}/bin/python"
RES=extra_eval/results
mkdir -p "$RES/render"
for d in ${DOMAINS:-3rscan multiscan}; do
    rd="${RENDER_RGBD_DIR:-eval_outputs/render_rgbd}/$d"
    for rule in ${RULES:-gt all}; do
        run="${RUN_PREFIX:-eval_outputs/render_}${d}_$rule"
        if [ "$rule" = gt ]; then  # GT-object marks: exact mark -> object id
            match=(--id_map "$rd/mark_sets.json")
            cp "$rd/mark_sets.json" "$RES/render/mark_sets_${d}_gt.json"
        else  # our Mask3D proposals: box-IoU matching as on ScanNet
            match=(--seg_attrs "$rd/mask3d/seg_attrs.pt" --gt_attrs "$rd/mask3d/gt_attrs.pt")
            cp "$rd/mask3d/mark_sets.json" "$RES/render/mark_sets_${d}_all.json"
        fi
        $PY extra_eval/beacon3d/score_beacon3d.py --domain "$d" "${match[@]}" \
            --preds_dir "$run" --anno_dir "$run/annos" --result_dir "$RES/beacon3d_${d}_render_$rule" \
            --qa_model "${QA_MODEL:-gpt-4o-mini-2024-07-18}" || exit 1
        cp "$rd/$rule/render_stats.json" "$RES/render/render_stats_${d}_$rule.json"
    done
done
echo SCORE_RENDER_RGBD_DONE
