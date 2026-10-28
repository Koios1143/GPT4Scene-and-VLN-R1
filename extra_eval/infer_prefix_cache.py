#!/usr/bin/env python3
"""GPT4Scene inference with one image-prefix KV cache per scene (``PREFIX_CACHE=1`` in run_infer.sh).

Upstream ``evaluate/infer.py`` feeds every question the full input: 32 marked frames + the
marked BEV image (about 8.5k visual tokens at 512 px) followed by the question. Only the
question text differs between the questions of a scene. Here the shared part is computed once:

* Questions of all ``--tasks`` are grouped by scene (each chunk / GPU gets whole scenes).
* Per scene, the vision tower and the LLM run once over the prefix, i.e. the chat template up
  to the last ``<|vision_end|>``. The post-RoPE keys / values of every layer are kept, one copy.
* Questions run in batches of ``--gen_batch``. Each row is the question suffix
  (``<prompt><|im_end|>\\n<|im_start|>assistant\\n``), right-padded, then greedy decoding.
  The attention of each layer is two flash-attn calls merged by log-sum-exp: the queries
  attend the shared prefix (``flash_attn_with_kvcache`` with ``cache_batch_idx`` = 0 for all
  rows, non-causal) and their own row's keys (a per-row buffer, causal, ``cache_seqlens`` =
  the row's real length, so padding is never attended and is overwritten by decoding).
* Positions are the M-RoPE ids of the full sequence: the prefix's from ``get_rope_index``,
  the text after it continues at ``max + 1`` on all three axes, as ``get_rope_index`` does.

Decoding is greedy (the checkpoint's generation_config: ``top_k=1``), at most 128 new tokens,
eos = ``<|im_end|>`` / ``<|endoftext|>``, as ``evaluate/qwen2_vl.py``. Prompts, images and
resizing come from upstream (``bulid_message``, ``process_vision_info``, the processor).
Output: ``<output_dir>/qwenvl_<task>/predictions_chunk_<i>.json`` in upstream's format, so
``score.sh`` reads both paths alike. An existing file is resumed (its qids are skipped).
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import torch
from flash_attn import flash_attn_func, flash_attn_with_kvcache
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evaluate"))
from qwen2_vl import bulid_message  # noqa: E402
from qwen_vl_utils import process_vision_info  # noqa: E402
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration  # noqa: E402
from transformers.models.qwen2_vl import modeling_qwen2_vl as mq  # noqa: E402

VISION_END = "<|vision_end|>"


class PrefixCacheContext:
    """Per-layer prefix keys / values (shared) and per-row suffix buffers for the patched attention."""

    def __init__(self):
        self.mode = None  # "prefix" while encoding the scene prefix, "rows" while answering
        self.pk, self.pv = {}, {}

    def start_rows(self, n_layers, bsz, cap, kv_heads, head_dim, dtype, device):
        self.sk = [torch.empty(bsz, cap, kv_heads, head_dim, dtype=dtype, device=device) for _ in range(n_layers)]
        self.sv = [torch.empty_like(t) for t in self.sk]
        self.seqlens = torch.zeros(bsz, dtype=torch.int32, device=device)
        self.prefix_batch_idx = torch.zeros(bsz, dtype=torch.int32, device=device)
        # per query row (an int would be sized by the cache's batch of 1)
        self.prefix_seqlens = torch.full((bsz,), self.pk[0].shape[1], dtype=torch.int32, device=device)


CTX = PrefixCacheContext()
_orig_forward = mq.Qwen2VLFlashAttention2.forward


def _attention_forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                       output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None):
    if CTX.mode is None:
        return _orig_forward(self, hidden_states, attention_mask, position_ids, past_key_value,
                             output_attentions, use_cache, cache_position, position_embeddings)
    bsz, q_len, _ = hidden_states.size()
    q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    cos, sin = position_embeddings
    q, k = mq.apply_multimodal_rotary_pos_emb(q, k, cos, sin, self.rope_scaling["mrope_section"])
    q, k, v = q.transpose(1, 2), k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()
    i = self.layer_idx
    if CTX.mode == "prefix":
        CTX.pk[i], CTX.pv[i] = k, v
        out = flash_attn_func(q, k, v, causal=True)
    else:
        o1, lse1 = flash_attn_with_kvcache(q, CTX.pk[i], CTX.pv[i], cache_seqlens=CTX.prefix_seqlens,
                                           cache_batch_idx=CTX.prefix_batch_idx, causal=False,
                                           return_softmax_lse=True)
        # appends k / v at each row's cache_seqlens (in place), then attends [0, seqlens + q_len)
        o2, lse2 = flash_attn_with_kvcache(q, CTX.sk[i], CTX.sv[i], k=k, v=v, cache_seqlens=CTX.seqlens,
                                           causal=True, return_softmax_lse=True)
        m = torch.maximum(lse1, lse2)
        w1 = torch.exp(lse1 - m).transpose(1, 2).unsqueeze(-1)  # (b, q, h, 1)
        w2 = torch.exp(lse2 - m).transpose(1, 2).unsqueeze(-1)
        out = ((o1.float() * w1 + o2.float() * w2) / (w1 + w2)).to(q.dtype)
    out = self.o_proj(out.reshape(bsz, q_len, self.hidden_size))
    return out, None, past_key_value


mq.Qwen2VLFlashAttention2.forward = _attention_forward


class SceneDataset(Dataset):
    """One item per scene: the processed prefix (images + template up to the last <|vision_end|>)
    and the tokenized suffix of every question."""

    def __init__(self, scenes, processor, resolution):
        self.scenes, self.processor, self.cfg = scenes, processor, argparse.Namespace(resolution=resolution)

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene_id, paths, items = self.scenes[idx]
        prefix, suffixes = None, []
        for it in items:
            text = self.processor.apply_chat_template(bulid_message(it["prompt"], self.cfg, None, paths),
                                                      tokenize=False, add_generation_prompt=True)
            cut = text.rindex(VISION_END) + len(VISION_END)
            assert prefix is None or text[:cut] == prefix, scene_id
            prefix = text[:cut]
            suffixes.append(self.processor.tokenizer(text[cut:], add_special_tokens=False)["input_ids"])
        image_inputs, _ = process_vision_info([bulid_message("", self.cfg, None, paths)])
        inputs = self.processor(text=[prefix], images=image_inputs, return_tensors="pt")
        return scene_id, items, dict(inputs), suffixes


@torch.no_grad()
def encode_prefix(model, inputs, device):
    input_ids = inputs["input_ids"].to(device)
    grid = inputs["image_grid_thw"].to(device)
    emb = model.model.embed_tokens(input_ids)
    img = model.visual(inputs["pixel_values"].to(device, model.visual.get_dtype()), grid_thw=grid)
    mask = (input_ids == model.config.image_token_id).unsqueeze(-1).expand_as(emb)
    emb = emb.masked_scatter(mask, img.to(emb.dtype))
    pos, _ = model.get_rope_index(input_ids, grid, None, torch.ones_like(input_ids))
    CTX.mode = "prefix"
    model.model(inputs_embeds=emb, position_ids=pos, use_cache=False)
    return int(pos.max()) + 1


@torch.no_grad()
def answer_batch(model, suffixes, next_pos, eos_ids, max_new_tokens, device):
    bsz = len(suffixes)
    lens = torch.tensor([len(s) for s in suffixes], device=device)
    s_max = int(lens.max())
    ids = torch.zeros(bsz, s_max, dtype=torch.long, device=device)
    for b, s in enumerate(suffixes):
        ids[b, :len(s)] = torch.tensor(s, device=device)
    cfg = model.config
    CTX.mode = "rows"
    CTX.start_rows(cfg.num_hidden_layers, bsz, s_max + max_new_tokens, cfg.num_key_value_heads,
                   cfg.hidden_size // cfg.num_attention_heads, model.dtype, device)
    pos = (next_pos + torch.arange(s_max, device=device)).view(1, 1, -1).expand(3, bsz, -1)
    h = model.model(inputs_embeds=model.model.embed_tokens(ids), position_ids=pos, use_cache=False)[0]
    last = h[torch.arange(bsz, device=device), lens - 1]
    CTX.seqlens.copy_(lens.to(torch.int32))
    out = torch.zeros(bsz, max_new_tokens, dtype=torch.long, device=device)
    done = torch.zeros(bsz, dtype=torch.bool, device=device)
    eos = torch.tensor(eos_ids, device=device)
    n = 0
    for t in range(max_new_tokens):
        tok = model.lm_head(last).argmax(-1)
        out[:, t] = tok
        n = t + 1
        done |= torch.isin(tok, eos)
        if bool(done.all()) or t == max_new_tokens - 1:
            break
        pos = (next_pos + CTX.seqlens.long()).view(1, -1, 1).expand(3, -1, 1)
        last = model.model(inputs_embeds=model.model.embed_tokens(tok[:, None]), position_ids=pos,
                           use_cache=False)[0][:, -1]
        CTX.seqlens += 1
    seqs = []
    for row in out[:, :n].tolist():
        cut = next((j for j, x in enumerate(row) if x in eos_ids), len(row))
        seqs.append(row[:cut])
    return seqs


def shard_scenes(scene_sizes, num_chunks, chunk_idx):
    """Whole scenes per chunk, largest first onto the least-loaded chunk."""
    load = [0] * num_chunks
    owner = {}
    for s, n in sorted(scene_sizes.items(), key=lambda x: (-x[1], x[0])):
        c = min(range(num_chunks), key=lambda j: (load[j], j))
        owner[s] = c
        load[c] += n
    return sorted(s for s, c in owner.items() if c == chunk_idx)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="config with val_file_dict (e.g. build_annos.py output)")
    ap.add_argument("--tasks", required=True, help="space- or #-separated val_file_dict keys")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--scene_anno", required=True, help="scene_id -> image paths (BEV + frames)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_chunks", type=int, default=1)
    ap.add_argument("--chunk_idx", type=int, default=0)
    ap.add_argument("--gen_batch", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_scenes", type=int, default=0, help="debug: only the first N scenes of this chunk")
    args = ap.parse_args()
    cfg = {}
    exec(open(args.config).read(), cfg)
    tasks = args.tasks.replace("#", " ").split()
    scene_anno = json.load(open(args.scene_anno))

    files, preds, done = {}, {}, {}
    by_scene = defaultdict(list)
    for task in tasks:
        out_dir = os.path.join(args.output_dir, f"qwenvl_{task}")
        os.makedirs(out_dir, exist_ok=True)
        files[task] = os.path.join(out_dir, f"predictions_chunk_{args.chunk_idx}.json")
        preds[task] = json.load(open(files[task])) if os.path.exists(files[task]) else []
        done[task] = {p["qid"] for p in preds[task]}
        for a in json.load(open(cfg["val_file_dict"][task])):
            by_scene[a["scene_id"]].append(dict(a, _task=task))
    my_scenes = shard_scenes({s: len(v) for s, v in by_scene.items()}, args.num_chunks, args.chunk_idx)
    if args.max_scenes:
        my_scenes = my_scenes[:args.max_scenes]
    scenes = []
    for s in my_scenes:
        todo = [a for a in by_scene[s] if a.get("qid", 0) not in done[a["_task"]]]
        if todo:
            scenes.append((s, scene_anno[s], todo))
    n_todo = sum(len(x[2]) for x in scenes)
    print(f"chunk {args.chunk_idx}/{args.num_chunks}: {len(my_scenes)} scenes, {n_todo} questions to run "
          f"({sum(len(v) for v in done.values())} resumed), tasks={tasks}, gen_batch={args.gen_batch}", flush=True)

    device = torch.device("cuda")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto", attn_implementation="flash_attention_2", device_map=None).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model_path)
    eos_ids = model.generation_config.eos_token_id
    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    assert model.generation_config.top_k == 1 or not model.generation_config.do_sample, "expects greedy decoding"

    loader = DataLoader(SceneDataset(scenes, processor, args.resolution), batch_size=None, shuffle=False,
                        num_workers=args.num_workers, prefetch_factor=2 if args.num_workers else None)
    t0, n_done, t_prefix = time.time(), 0, 0.0
    for si, (scene_id, items, inputs, suffixes) in enumerate(loader):
        tp = time.time()
        next_pos = encode_prefix(model, inputs, device)
        torch.cuda.synchronize()
        t_prefix += time.time() - tp
        order = sorted(range(len(items)), key=lambda j: -len(suffixes[j]))
        for b0 in range(0, len(order), args.gen_batch):
            idx = order[b0:b0 + args.gen_batch]
            seqs = answer_batch(model, [suffixes[j] for j in idx], next_pos, eos_ids, args.max_new_tokens, device)
            texts = processor.batch_decode(seqs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for j, text in zip(idx, texts):
                a = items[j]
                type_info = a.get("sqa_type", a.get("eval_type", a.get("type_info", 0)))
                preds[a["_task"]].append({
                    "scene_id": scene_id, "qid": a.get("qid", 0), "gt_id": int(a.get("obj_id", 0)),
                    "pred_id": int(a.get("pred_id", 0)), "prompt": a["prompt"], "pred": text,
                    "ref_captions": a.get("ref_captions", []), "type_info": type_info})
        CTX.mode, CTX.pk, CTX.pv = None, {}, {}
        n_done += len(items)
        for task in {a["_task"] for a in items}:
            json.dump(preds[task], open(files[task], "w"), indent=4)
        el = time.time() - t0
        print(f"[{si + 1}/{len(scenes)}] {scene_id}: {len(items)} q, {n_done}/{n_todo} done, "
              f"{el / 60:.1f} min ({n_done / el:.2f} q/s, prefix {t_prefix:.0f}s)", flush=True)
    json.dump({"chunk_idx": args.chunk_idx, "scenes": len(scenes), "questions": n_done,
               "seconds": round(time.time() - t0, 1), "prefix_seconds": round(t_prefix, 1),
               "gen_batch": args.gen_batch, "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)},
              open(os.path.join(args.output_dir, f"prefix_cache_stats_chunk_{args.chunk_idx}.json"), "w"), indent=1)
    print("PREFIX_CACHE_DONE", flush=True)


if __name__ == "__main__":
    main()
