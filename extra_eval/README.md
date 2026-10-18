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

### Beacon3D, ScanNet (12 of 16 scenes)

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
