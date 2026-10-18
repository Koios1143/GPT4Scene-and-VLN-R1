#!/bin/bash
# GPT4Scene inference/scoring env as a uv venv, ported to current GPUs: the README pins
# torch 2.5.0+cu121, which has no kernels for Blackwell (sm_120), so this uses
# torch 2.7.1+cu128 + the flash-attn 2.8.3 prebuilt wheel, transformers 4.46.1 (the
# upper bound of requirements.txt), qwen-vl-utils, pycocoevalcap (Scan2Cap) and openai
# (Beacon3D's GPT-judged QA evaluator).
#   bash extra_eval/build_env.sh
# Run on a GPU node with internet access. Paths come from env.sh (see env.sh.example):
#   GPT4SCENE_ENV   where the venv goes   (default .venv, under the repo root)
#   UV              the uv binary         (default: uv on PATH)
#   UV_CACHE_DIR / TMPDIR   point these at a disk with room if $HOME is full
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f env.sh ] && source env.sh
unset CONDA_PREFIX CONDA_DEFAULT_ENV
UV="${UV:-uv}"
ENV="${GPT4SCENE_ENV:-.venv}"
$UV python install 3.10
[ -x "$ENV/bin/python" ] || $UV venv "$ENV" -p 3.10 --seed
PY="$ENV/bin/python"
PIP="$UV pip install --python $PY"
$PIP torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
$PIP transformers==4.46.1 accelerate==1.0.1 "numpy<2" scipy pyyaml termcolor tqdm pillow \
    qwen-vl-utils==0.0.8 pycocoevalcap openai
$PIP "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
$PY -c "
import torch, transformers, flash_attn
from flash_attn import flash_attn_func
print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
print('transformers', transformers.__version__, 'flash_attn', flash_attn.__version__)
q = torch.randn(1, 16, 8, 128, device='cuda', dtype=torch.bfloat16)
print('flash-attn kernel ok', flash_attn_func(q, q, q).shape)
"
echo ENV_DONE
