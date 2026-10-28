# Extra benchmarks for GPT4Scene

The released checkpoint
[`alexzyqi/GPT4Scene-qwen2vl_full_sft_mark_32_3D_img512`](https://huggingface.co/alexzyqi/GPT4Scene-qwen2vl_full_sft_mark_32_3D_img512),
evaluated on **ScanRefer, Multi3DRefer, Scan2Cap, ScanQA, SQA3D (val + test), Real-3DQA** and
**Beacon3D (ScanNet, 3RScan, MultiScan)** with the repo's own model code and scorers
(32 marked frames + marked BEV image at 512 px, greedy, 128 new tokens).

## Run

```bash
cp env.sh.example env.sh            # fill in paths (+ OPENAI_API_KEY for Beacon3D QA)
bash extra_eval/build_env.sh        # uv venv: torch 2.7.1+cu128, transformers 4.46.1, flash-attn 2.8.3
bash extra_eval/download.sh         # checkpoint, GPT4Scene-Val-Annotation, GPT4Scene-All val images
PREFIX_CACHE=1 bash extra_eval/run_infer.sh   # GPU; one process per visible GPU; all benchmarks
                                              # (drop PREFIX_CACHE=1 for the upstream evaluate/infer.py path)
bash extra_eval/score.sh            # CPU; Java for pycocoevalcap (Scan2Cap / ScanQA / SQA3D)

# Beacon3D on re-rendered inputs (needs the 16 scenes' ScanNet .sens, see below)
bash extra_eval/render/run_render.sh                     # frames, mark sets, renders per rule
for r in released gt all; do
    SCENE_ANNO=$RENDER_DIR/$r/selected_images.json OUT=eval_outputs/render_$r \
        TASKS="beacon3d_scannet_grounding beacon3d_scannet_qa" bash extra_eval/run_infer.sh
done
bash extra_eval/render/score_render.sh

# Beacon3D 3RScan / MultiScan (raw scans: RSCAN_RAW_DIR / MULTISCAN_RAW_DIR, see env.sh.example)
for d in 3rscan multiscan; do
    DOMAIN=$d bash extra_eval/render/run_render_rgbd.sh      # frames, meshes, rule gt
    DOMAIN=$d bash extra_eval/render/run_mask3d_rgbd.sh      # Mask3D proposals, rule all
    for r in gt all; do
        DOMAIN=$d SCENE_ANNO=$RENDER_RGBD_DIR/$d/$r/selected_images.json OUT=eval_outputs/render_${d}_$r \
            TASKS="beacon3d_${d}_grounding beacon3d_${d}_qa" bash extra_eval/run_infer.sh
    done
done
bash extra_eval/render/score_render_rgbd.sh
```

The README's torch 2.5.0+cu121 has no Blackwell (sm_120) kernels, hence the newer stack;
the model code is the repo's, unchanged.

## Results

### Paper benchmarks (ScanNet val, Mask3D proposals)

Scored by `score_extra.py` with the repo's own scorers (`evaluate/utils/eval.py`, as
`evaluate/calculate_scores.py` calls them). The released checkpoint is the paper's
**GPT4Scene-HDM** (arXiv 2501.01428, Table 1); paper numbers in parentheses.
ScanRefer / Multi3DRefer / ScanQA / SQA3D: `PREFIX_CACHE=1` run; Scan2Cap: upstream path
(the fast path gives CIDEr@0.5 85.76, see *Faster evaluation*). All in [`results/`](results)
(`<bench>/scores.json` + `preds.json`, `summary.json`).

| Benchmark | Metric | Ours | Paper |
|---|---|---:|---:|
| ScanRefer (9508) | Acc@0.25 / Acc@0.5 | 62.33 / 56.75 | 62.6 / 57.0 |
| Multi3DRefer (11120) | F1@0.25 / F1@0.5 | 64.14 / 59.50 | 64.5 / 59.8 |
| Scan2Cap (2007) | CIDEr@0.5 / B-4@0.5 | 85.18 / 40.25 | 86.3 / 40.6 |
| ScanQA val (4675) | CIDEr / B-1 / B-4 / METEOR / ROUGE / EM | 96.56 / 44.44 / 16.13 / 18.84 / 46.45 / 28.17 | 96.3 / 44.4 / 15.5 / 18.9 / 46.5 / 28.2 |
| SQA3D val (3261) | EM / EM-R | 59.34 / 62.50 | 60.6 / 63.3 |
| SQA3D test (3519) | EM / EM-R | 60.44 / 63.46 | – |

Scan2Cap at IoU 0.25: CIDEr 90.52, B-4 42.73, METEOR 29.22, ROUGE-L 61.76 (@0.5: 28.18 / 59.16).

**SQA3D splits.** GPT4Scene releases and reports SQA3D *val* only (`sqa3d_val.json`). The
standard test split (3519 questions, 67 scenes, all with released images) is added from
Chat-Scene's `sqa3d_test.json` in the same prompt format (`build_annos.py`). Per question type
(What / Is / How / Can / Which / Others), test EM: 54.14 / 73.93 / 60.65 / 70.71 / 47.86 / 59.19.
(BLEU on one-word answers is degenerate in this scorer and not reported.)

### Real-3DQA (situated QA debiased + viewpoint rotation)

HF `Oliver-Ma/Real-3DQA`, scored by its official `evaluate_debiased_exact_match.py` and
`evaluate_rotation_robustness.py` (vendored in `real3dqa/`, unmodified). Questions are posed in
GPT4Scene's SQA3D format (`<situation> <question> Answer the question using a single word or
phrase.`); the rotation splits change only the situation text, so the scene input is the same
as for SQA3D. [`results/real3dqa/`](results/real3dqa)

| Split | EM | EM-R |
|---|---:|---:|
| debiased_test (1485) | 31.45 | 36.57 |
| rotation 0 / 90 / 180 / 270 (750 each) | 14.00 / 13.20 / 13.87 / 15.47 | 17.47 / 16.13 / 17.87 / 19.20 |

| Rotation consistency | One | Two | Three | Four | VRS |
|---|---:|---:|---:|---:|---:|
| | 54.93 | 13.20 | 2.13 | 0.40 | 17.67 |

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

### Beacon3D, 3RScan and MultiScan — re-rendered inputs

No GPT4Scene images or Mask3D proposals exist for these domains, so both are produced here
(`render/*_rgbd.py`) from the raw scans: 3RScan through its official download script
(`sequence.zip` RGB-D frames, textured mesh, instance labels), MultiScan from the gated HF
dataset `3dlg-hcvc/MultiScan` (mp4 + depth stream + ARKit poses, mesh with per-face object ids).
Frames are sampled as on ScanNet (`linspace(0, N-1, 64)`), turned upright from the camera
poses (3RScan's Tango frames are stored sideways), and marked exactly like the ScanNet renders.
Pose/mesh frames were checked by back-projecting depth onto the mesh (median 1-4 cm).

* **GT objects** (`gt`): the GT instances themselves are marked (ids 0..N-1 over non-structure
  objects), so grounding is scored exactly (mark id -> Beacon3D object id) and the two
  grounding variants coincide — the GT-mask condition (Chat-Scene used GT masks on these domains too).
* **All proposals** (`all`): our run of Mask3D with the ScanNet200 checkpoint GPT4Scene's marks
  come from (`render/mask3d_infer.py`, Mask3D's own test + export with the official ScanNet200
  settings), top-100 by score as in Chat-Scene's `mask3d_inst_seg`, then the same `all` rule as
  on ScanNet; grounding by box IoU as on ScanNet. Checked first on ScanNet: on scene0164_00 /
  scene0011_00 it reproduces Chat-Scene's released top-100 proposals (100% / 98% same rank,
  every released proposal matched at mask IoU 1.0; [`results/render/mask3d_scannet_check.json`](results/render/mask3d_scannet_check.json)).
  Being trained on ScanNet, it covers fewer Beacon3D targets here: a marked proposal with
  IoU >= 0.25 exists for 76.0% (3RScan) / 71.5% (MultiScan) of the grounding targets.

All three domains, both rules (ScanNet rows repeated from above; QA judge `gpt-4o-mini-2024-07-18`):

| Domain (scenes) | Mark rule | G case | G case @0.25 | app. | geo. | spatial | func. | chain | G object | G object @0.25 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ScanNet (16) | GT objects | 66.05 | 66.05 | 64.72 | 60.27 | 62.53 | 66.67 | 59.83 | 52.47 | 52.47 |
| ScanNet (16) | All proposals | 64.58 | 60.18 | 64.72 | 59.82 | 60.99 | 55.56 | 57.92 | 49.87 | 45.71 |
| 3RScan (9) | GT objects | 45.22 | 45.22 | 42.80 | 36.41 | 37.96 | – | 34.19 | 26.84 | 26.84 |
| 3RScan (9) | All proposals | 33.03 | 27.88 | 32.92 | 24.62 | 26.75 | – | 27.21 | 23.90 | 20.96 |
| MultiScan (5) | GT objects | 51.90 | 51.90 | 44.16 | 28.12 | 46.98 | – | 40.88 | 35.22 | 35.22 |
| MultiScan (5) | All proposals | 42.92 | 35.92 | 33.77 | 40.62 | 40.72 | – | 33.12 | 29.56 | 23.27 |

| Domain (scenes) | Mark rule | QA GPT-score | QA object | class | app. | geo. | spatial | exist. | func. | EM | EM-R | R1 | R2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ScanNet (16) | GT objects | 62.23 | 21.30 | 47.07 | 68.24 | 61.23 | 55.41 | 69.80 | 83.33 | 59.05 | 61.30 | 40.64 | 43.58 |
| ScanNet (16) | All proposals | 60.30 | 18.70 | 43.69 | 67.87 | 60.76 | 54.92 | 63.67 | 83.33 | 56.80 | 59.31 | 37.56 | 45.82 |
| 3RScan (9) | GT objects | 52.82 | 8.82 | 42.93 | 57.84 | 56.36 | 51.97 | 48.82 | – | 49.63 | 53.55 | 24.30 | 71.39 |
| 3RScan (9) | All proposals | 51.65 | 6.99 | 44.02 | 55.91 | 53.41 | 51.31 | 48.34 | – | 48.28 | 52.21 | 24.77 | 75.72 |
| MultiScan (5) | GT objects | 37.32 | 4.40 | 27.82 | 47.26 | 50.00 | 30.15 | 36.44 | – | 33.75 | 35.85 | 33.68 | 68.79 |
| MultiScan (5) | All proposals | 37.32 | 5.03 | 27.42 | 44.89 | 50.00 | 31.25 | 38.14 | – | 33.33 | 35.64 | 23.96 | 70.93 |

(3RScan / MultiScan have no functionality cases; 3RScan 1267 / 816, MultiScan 657 / 477
grounding / QA cases.) Results: `results/beacon3d_{3rscan,multiscan}_render_{gt,all}/`.

### Beacon3D, ScanNet — released images (12 of 16 scenes, reference)

Beacon3D's evaluators (vendored in `beacon3d/Beacon3D`, unmodified), QA judged by
**`gpt-4o-mini-2024-07-18`** (upstream default is `gpt-4o-2024-11-20`).
[`results/beacon3d_scannet/`](results/beacon3d_scannet)

**Coverage.** GPT4Scene only sees a scene through its released marked frames
(`selected_images_mark_3D_val_32.json`, 143 ScanNet val scenes). Four Beacon3D scenes have
none: `scene0458_01`, `scene0568_02`, `scene0591_02`, `scene0697_01`. They are left out and
the official evaluators run on Beacon3D's data + metadata filtered to the other 12 scenes:
**1254 / 1611 grounding cases, 894 / 1155 QA cases** (3RScan / MultiScan: re-rendered, above).

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

## Faster evaluation: `PREFIX_CACHE=1`

Every question of a scene shares the same ~8.5k-token image prefix (32 frames + BEV), and the
question text comes after it. `infer_prefix_cache.py` therefore runs all tasks in one pass
grouped by scene (whole scenes sharded over GPUs, balanced by question count): the vision tower
and the LLM run once per scene over the chat template up to the last `<|vision_end|>`, and the
per-layer post-RoPE K/V of that prefix are kept. Questions are then decoded greedily in batches
of 32 (right-padded suffixes): a patched `Qwen2VLFlashAttention2.forward` attends to the shared
prefix (`flash_attn_with_kvcache`, non-causal) and to the row's own keys (causal) and merges the
two by log-sum-exp; M-RoPE positions continue at prefix max + 1. Prompts, images, greedy
decoding, 128 new tokens and the output files are those of `evaluate/infer.py`.

Checked against the upstream path on the same inputs
([`results/prefix_cache/agreement.json`](results/prefix_cache/agreement.json)): identical
answers on 95–99% of Beacon3D grounding / QA items in every domain and mark rule (97.8% on
released-images ScanNet); Scan2Cap captions match exactly on 76% (bf16 near-ties early in a
long caption), CIDEr@0.5 85.18 -> 85.76. Scores (QA judge `gpt-4o-mini-2024-07-18`):

| Beacon3D run | G case (upstream / fast) | QA GPT-score | QA EM |
|---|---|---|---|
| ScanNet released images (12 sc.) | 59.84 / 60.08 | 60.12 / 60.01 | 56.15 / 56.38 |
| ScanNet render `gt` | 66.05 / 66.05 | 62.23 / 62.66 | 59.05 / 59.74 |
| ScanNet render `all` | 64.58 / 64.58 | 60.30 / 60.11 | 56.80 / 56.62 |
| 3RScan render `gt` / `all` | 45.22 / 45.38, 33.03 / 32.46 | 52.82 / 53.00, 51.65 / 52.85 | 49.63 / 49.63, 48.28 / 49.26 |
| MultiScan render `gt` / `all` | 51.90 / 51.60, 42.92 / 43.23 | 37.32 / 38.84, 37.32 / 38.94 | 33.75 / 35.22, 33.33 / 35.01 |

(MultiScan has 477 QA items, ~0.2 per item.) The tables above keep the upstream-path numbers
for Scan2Cap and Beacon3D; the other benchmarks were run with `PREFIX_CACHE=1` only.

Wall clock on 2 Blackwell GPUs (upstream = Slurm job time, fast = `run_infer.sh` incl. model load):

| Run | upstream | `PREFIX_CACHE=1` |
|---|---:|---:|
| Scan2Cap + Beacon3D-ScanNet, released images (4155 items) | 23.4 min | 3.8 min (6×) |
| Beacon3D-ScanNet render, one rule (2766 items) | 37.7 min | 26 s |
| Beacon3D-3RScan / MultiScan render, one rule (2083 / 1134 items) | 19.3 / 18.3 min | 18 / 16 s |
| all benchmarks on released images (40723 items) | – | 6.4 min |

Peak GPU memory 17.4 GB per process. The gain is largest for many short answers per scene
(Beacon3D: ~170 questions per scene, one prefix each) and smallest for long captions.

## Dependencies: `deps/`

All data, checkpoints and environments a run reads are reached through the gitignored
`deps/` (hard links or copies, or symlinks only into storage owned by this baseline), and
`env.sh` points there, so deleting a shared copy elsewhere does not break re-runs. The venv's
interpreter is a copy too (`pyvenv.cfg` home -> `deps/uv-python-3.10.22`).

## Files

* `build_env.sh`, `download.sh`, `run_infer.sh`, `score.sh` — the steps above; paths from `env.sh`
* `infer_prefix_cache.py` — `PREFIX_CACHE=1` inference (see *Faster evaluation*);
  `compare_preds.py` — item agreement of two runs
* `build_annos.py` — SQA3D test + Real-3DQA -> GPT4Scene val annotations + config
* `score_extra.py` — ScanRefer / Multi3DRefer / Scan2Cap / ScanQA / SQA3D / Real-3DQA scoring;
  `real3dqa/` — the official Real-3DQA evaluators (vendored, see `real3dqa/VENDORED.md`)
* `beacon3d/build_annos.py` — Beacon3D -> GPT4Scene val annotations (its own ScanRefer/ScanQA prompts) + config
* `beacon3d/score_beacon3d.py` — proposal->GT matching, subset data/metadata, official evaluators, `scores.json`
* `beacon3d/beacon_grounding_shim.py` — runs `evaluate_grounding.py` skipping knowledge tags with no cases
* `render/` — re-rendering GPT4Scene's inputs (see *Renderer* above); `*_rgbd.py` +
  `run_render_rgbd.sh` for 3RScan / MultiScan; `mask3d_infer.py` (+ `mask3d_shims/`, a
  re-implementation of the three detectron2 functions Mask3D's loss imports),
  `prepare_mask3d_rgbd.py`, `mask3d_to_proposals.py`, `compare_mask3d.py`, `run_mask3d_rgbd.sh`
