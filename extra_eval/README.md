# Extra benchmarks for GPT4Scene

The released checkpoint
[`alexzyqi/GPT4Scene-qwen2vl_full_sft_mark_32_3D_img512`](https://huggingface.co/alexzyqi/GPT4Scene-qwen2vl_full_sft_mark_32_3D_img512),
evaluated on **Scan2Cap** and **Beacon3D (ScanNet)** with the repo's own inference
(`evaluate/infer.py`, 32 marked frames + marked BEV image at 512 px, greedy, 128 new tokens).

## Run

```bash
cp env.sh.example env.sh            # fill in paths (+ OPENAI_API_KEY for Beacon3D QA)
bash extra_eval/build_env.sh        # uv venv: torch 2.7.1+cu128, transformers 4.46.1, flash-attn 2.8.3
bash extra_eval/download.sh         # checkpoint, GPT4Scene-Val-Annotation, GPT4Scene-All val images
bash extra_eval/run_infer.sh        # GPU; one process per visible GPU
bash extra_eval/score.sh            # CPU; Java for Scan2Cap's METEOR/PTB tokenizer

# Beacon3D on re-rendered inputs (needs the 16 scenes' ScanNet .sens, see below)
bash extra_eval/render/run_render.sh                     # frames, mark sets, renders per rule
for r in released gt all; do
    SCENE_ANNO=$RENDER_DIR/$r/selected_images.json OUT=eval_outputs/render_$r \
        TASKS="beacon3d_scannet_grounding beacon3d_scannet_qa" bash extra_eval/run_infer.sh
done
bash extra_eval/render/score_render.sh
```

The README's torch 2.5.0+cu121 has no Blackwell (sm_120) kernels, hence the newer stack;
the model code is the repo's, unchanged.

## Results

### Scan2Cap (ScanNet val, 2007 objects, Mask3D proposals)

Scored by the repo's `evaluate/calculate_scores.py` (`calc_scan2cap_score`).
[`results/scan2cap/`](results/scan2cap)

| IoU  | CIDEr | BLEU-4 | METEOR | ROUGE-L |
|------|------:|-------:|-------:|--------:|
| 0.25 | 90.52 | 42.73  | 29.22  | 61.76   |
| 0.50 | 85.18 | 40.25  | 28.18  | 59.16   |

### Beacon3D, ScanNet — all 16 scenes on re-rendered inputs (main result)

GPT4Scene's released images cover only 12 of the 16 scenes, and they mark only some objects —
chosen with GT and another benchmark's annotations (see *Why re-render*), so Beacon3D targets
outside that set cannot be named at all. We therefore re-render GPT4Scene's inputs for all 16
scenes from ScanNet (`render/`) under two annotation-agnostic mark rules and report both:

* **GT objects** (`gt`): every GT object except wall/floor/ceiling is marked with its best
  box-IoU Mask3D proposal (IoU >= 0.25) — the condition of a GT-mask model such as Chat-Scene.
* **All proposals** (`all`): every Mask3D proposal, no GT (minus predicted wall/floor/ceiling
  and duplicate boxes) — the condition of a detector-only model such as Grounded 3D-LLM.

Official Beacon3D evaluators, QA judged by **`gpt-4o-mini-2024-07-18`**; 1611 grounding /
1155 QA cases. Grounding: the predicted proposal is matched to the GT object of highest box
IoU, correct if that is a target and IoU > 0 (*any overlap*) or IoU >= 0.25.

| Mark rule | G case | G case @0.25 | app. | geo. | spatial | func. | chain | G object | G object @0.25 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GT objects    | 66.05 | 66.05 | 64.72 | 60.27 | 62.53 | 66.67 | 59.83 | 52.47 | 52.47 |
| All proposals | 64.58 | 60.18 | 64.72 | 59.82 | 60.99 | 55.56 | 57.92 | 49.87 | 45.71 |

| Mark rule | QA GPT-score | QA object | class | app. | geo. | spatial | exist. | func. | EM | EM-R | R1 | R2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GT objects    | 62.23 | 21.30 | 47.07 | 68.24 | 61.23 | 55.41 | 69.80 | 83.33 | 59.05 | 61.30 | 40.64 | 43.58 |
| All proposals | 60.30 | 18.70 | 43.69 | 67.87 | 60.76 | 54.92 | 63.67 | 83.33 | 56.80 | 59.31 | 37.56 | 45.82 |

(R1 / R2 use the any-overlap grounding; the @0.25 chain numbers are in each `scores.json`.)
[`results/beacon3d_scannet_render_gt/`](results/beacon3d_scannet_render_gt),
[`results/beacon3d_scannet_render_all/`](results/beacon3d_scannet_render_all)

**Renderer validation.** Rendering the 12 covered scenes with a reconstruction of the released
rule (`released`, below) and running the same model gives, against the released images on the
same 1254 / 894 items: grounding (item-level accuracy, not Beacon3D's deduplicated case score) 60.53 -> 60.45 (item agreement 91%), QA GPT-score 60.12 -> 60.15 (agreement
94%), EM 56.15 -> 56.71 ([`results/render/validation_12scenes.json`](results/render/validation_12scenes.json);
the three scenes we first checked, `validation.json`, agree within ~3 points). The same rule on
all 16 scenes: grounding case 58.20, QA GPT-score 61.77
([`results/beacon3d_scannet_render_released/`](results/beacon3d_scannet_render_released)).

#### Why re-render

The released marks are not all 100 Mask3D proposals. Reading the ids off the released BEV
images of scene0011_00 / 0025_00 / 0164_00, all 70 are reproduced (plus 2 extra) by: the best
box-IoU proposal of each GT object referenced in **ScanRefer / Multi3DRefer val**, plus that of
every other non-structure GT object whose best proposal has IoU >= 0.8 (`render/mark_sets.py`,
rule `released`). On ScanRefer every target is marked by construction; on Beacon3D (12
covered scenes) 210 / 1254 targets are not, and GPT4Scene gets 4.3% of those right vs 71.8% of
the marked ones. The authors confirm the preprocessing uses GT and is unreleased (upstream
issues #2, #19: BEV rendered in Blender).

#### Renderer (`render/`)

* `extract_frames.py` — frames `linspace(0, N-1, 64)` of the ScanNet `.sens` (identical to the
  released file names), the model gets the even 32 + BEV; released frames = raw frame + marks.
* `mark_sets.py` — the three rules (`released`, `gt`, `all`).
* `render_marks.py` — 2D marks: disk BGR (0,0,137) r=30 px + white Hershey digits at the image
  centroid of the proposal's depth-visible mesh vertices (>= 1, which best matches the
  released per-frame mark counts); BEV: perspective top-down render of the aligned mesh,
  ceiling removed, "BEV Map" band, red ids on leader lines. Mark masks = Chat-Scene's
  `mask3d_inst_seg` (its boxes equal GPT4Scene's `scannet_mask3d_val_attributes.pt` exactly).
* `run_render.sh`, `score_render.sh`, `compare_validation.py`; inference = `run_infer.sh`
  with `SCENE_ANNO=<render>/<rule>/selected_images.json`.

ScanNet `.sens` files are needed for the 16 scenes (ScanNet's `download-scannet.py --type .sens`,
under the ScanNet Terms of Use; not redistributed here).

### Beacon3D, ScanNet — released images (12 of 16 scenes, reference)

Beacon3D's evaluators (vendored in `beacon3d/Beacon3D`, unmodified), QA judged by
**`gpt-4o-mini-2024-07-18`** (upstream default is `gpt-4o-2024-11-20`).
[`results/beacon3d_scannet/`](results/beacon3d_scannet)

**Coverage.** GPT4Scene only sees a scene through its released marked frames
(`selected_images_mark_3D_val_32.json`, 143 ScanNet val scenes). Four Beacon3D scenes have
none: `scene0458_01`, `scene0568_02`, `scene0591_02`, `scene0697_01`. They are left out and
the official evaluators run on Beacon3D's data + metadata filtered to the other 12 scenes:
**1254 / 1611 grounding cases, 894 / 1155 QA cases**. 3RScan and MultiScan are not
evaluated (no RGB video to render GPT4Scene's inputs from).

**Grounding.** GPT4Scene answers with the `<OBJxxx>` mark of a Mask3D proposal. The first
valid mark (as in its ScanRefer eval; all 1254 answers had one) is matched to the ScanNet GT
object of highest 3D box IoU; correct = that object is a target and IoU > 0 (*any overlap*)
or IoU >= 0.25 (*IoU@0.25*).

| Grounding | case | class | app. | geo. | spatial | func. | chain | object |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| any overlap | 59.84 | 59.84 | 57.50 | 52.94 | 57.14 | 22.22 | 55.37 | 47.32 |
| IoU@0.25    | 58.93 | 58.93 | 56.79 | 50.00 | 56.33 | 22.22 | 54.25 | 46.64 |

| QA | GPT-score (case) | object | class | app. | geo. | spatial | exist. | func. | EM | EM-R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| | 60.12 | 18.79 | 43.13 | 64.36 | 59.35 | 57.13 | 66.49 | 83.33 | 56.15 | 59.06 |

Grounding->QA chain (Beacon3D's `eval_chain`; G1Q1 / G1Q0 / G0Q1 / G0Q0):

| grounding variant | w/o extra | overall | R1 (higher better) | R2 (lower better) |
|---|---|---|---:|---:|
| any overlap | 33.66 / 12.33 / 30.14 / 23.87 | 30.09 / 17.23 / 29.75 / 22.93 | 34.05 | 49.72 |
| IoU@0.25    | 33.27 / 12.13 / 30.53 / 24.07 | 29.75 / 16.89 / 30.09 / 23.27 | 33.51 | 50.28 |

## Files

* `build_env.sh`, `download.sh`, `run_infer.sh`, `score.sh` — the steps above; paths from `env.sh`
* `beacon3d/build_annos.py` — Beacon3D -> GPT4Scene val annotations (its own ScanRefer/ScanQA prompts) + config
* `beacon3d/score_beacon3d.py` — proposal->GT matching, subset data/metadata, official evaluators, `scores.json`
* `beacon3d/beacon_grounding_shim.py` — runs `evaluate_grounding.py` skipping knowledge tags with no cases
* `render/` — re-rendering GPT4Scene's inputs (see *Renderer* above)
