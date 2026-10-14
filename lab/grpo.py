"""Bước 9 (E5): GRPO với reward IoU, khởi tạo từ adapter SFT tốt nhất.

  python -m lab.grpo --data data --out out --init-adapter <dir> [--init-projector <pt>] [--impl trl|manual]

Chỉ cập nhật LoRA của LLM; projector (nếu có) giữ cố định ở trọng số SFT.
Ảnh huấn luyện được lọc còn <= 4 box để đáp án vừa max_completion_length=128 token
(một box ≈ 22 token): nếu không, completion bị cắt -> JSON hỏng -> reward 0 cho mọi
mẫu của ảnh đông box, và GRPO sẽ học cách "nói ít box" (một dạng reward hacking).
"""
import argparse
import gc
import glob
import json
import math
import random
import time
from pathlib import Path

import torch
from PIL import Image

from lab.labels import get_prompt
from lab.reward import compute_reward, gt_from_objects
from lab.run import predict
from lab.vlm import load_trained
from scripts.labcommon import read_jsonl
from scripts.score import parse_output

CFG = dict(num_generations=4, max_completion_length=128, max_steps=60, learning_rate=5e-6,
           beta=0.0, temperature=1.0, prompts_per_step=2)


def _find(path, pattern):
    """Đường dẫn trực tiếp, hoặc tìm trong /kaggle/input (output của kernel SFT gắn làm input)."""
    if path and Path(path).exists():
        return str(path)
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    return str(Path(hits[0]).parent if pattern.endswith(".json") else hits[0]) if hits else None


def _text(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c and isinstance(c[-1], dict):  # dạng hội thoại
        cont = c[-1].get("content")
        if isinstance(cont, list):
            return "".join(x.get("text", "") for x in cont if isinstance(x, dict))
        return cont or ""
    return str(c)


class HackStats:
    """Thống kê completion để phát hiện reward hacking."""

    def __init__(self, path):
        self.path, self.rows = path, []

    def add(self, completions, rewards, gts):
        n = len(completions)
        parsed = [parse_output(c) for c in completions]
        valid = [p for p in parsed if p is not None]
        giant = sum(1 for p in valid for b in p
                    if (b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1]) > 0.25 * 1e6)
        row = {
            "reward_mean": sum(rewards) / n,
            "reward_std": (sum((r - sum(rewards) / n) ** 2 for r in rewards) / n) ** 0.5,
            "invalid_frac": 1 - len(valid) / n,
            "empty_frac": sum(1 for p in valid if not p) / n,
            "empty_when_gt_frac": sum(1 for p, g in zip(parsed, gts) if p == [] and g) / n,
            "pred_boxes_mean": sum(len(p) for p in valid) / max(1, len(valid)),
            "gt_boxes_mean": sum(len(g) for g in gts) / n,
            "giant_boxes": giant,
            "completion_chars": sum(len(c) for c in completions) / n,
        }
        self.rows.append(row)
        return row


def _entries(data, max_boxes=4, seed=0):
    tr = [e for e in read_jsonl(Path(data) / "splits" / "train.jsonl") if len(e["objects"]) <= max_boxes]
    random.Random(seed).shuffle(tr)
    return tr


# ---------------------------------------------------------------- TRL GRPOTrainer
def run_trl(model, processor, data, out, entries, log):
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    n_prompts = CFG["max_steps"] * CFG["prompts_per_step"]
    rows = []
    for e in entries[:n_prompts]:
        rows.append({
            "prompt": [{"role": "user", "content": [{"type": "image"},
                                                    {"type": "text", "text": get_prompt()}]}],
            "image": Image.open(Path(data) / e["image_path"]).convert("RGB"),
            "gt": json.dumps(gt_from_objects(e["objects"])),
        })
    ds = Dataset.from_list(rows)
    stats = HackStats(out / "grpo_hack_stats.jsonl")

    def iou_reward(prompts, completions, gt, **kw):
        texts = [_text(c) for c in completions]
        gts = [[{"label": g["label"], "bbox": tuple(g["bbox"])} for g in json.loads(x)] for x in gt]
        rewards = [compute_reward(t, g) for t, g in zip(texts, gts)]
        log({"kind": "batch", **stats.add(texts, rewards, gts), "sample": texts[0][:200]})
        return rewards

    G = CFG["num_generations"]
    import dataclasses
    kw = dict(
        output_dir=str(out / "grpo_ckpt"), max_steps=CFG["max_steps"],
        learning_rate=CFG["learning_rate"], beta=CFG["beta"], num_generations=G,
        max_completion_length=CFG["max_completion_length"], temperature=CFG["temperature"],
        per_device_train_batch_size=G, gradient_accumulation_steps=CFG["prompts_per_step"],
        lr_scheduler_type="constant", logging_steps=1, save_strategy="no",
        fp16=True, bf16=False, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False, report_to="none", use_vllm=False, seed=42,
    )
    fields = {f.name for f in dataclasses.fields(GRPOConfig)}
    print("GRPOConfig bỏ qua:", sorted(set(kw) - fields), flush=True)
    args = GRPOConfig(**{k: v for k, v in kw.items() if k in fields})
    trainer = GRPOTrainer(model=model, args=args, reward_funcs=[iou_reward],
                          train_dataset=ds, processing_class=processor)
    # GRPOTrainer đưa adapter về bf16; T4 + GradScaler fp16 chỉ unscale được grad fp32.
    casted = {}
    for n, p in trainer.model.named_parameters():
        if p.requires_grad and p.dtype != torch.float32:
            casted[str(p.dtype)] = casted.get(str(p.dtype), 0) + 1
            p.data = p.data.float()
    print("ép về fp32:", casted, flush=True)
    trainer.train()
    for h in trainer.state.log_history:
        log({"kind": "trl", **{k: v for k, v in h.items() if isinstance(v, (int, float))}})
    return trainer.model


# ---------------------------------------------------------------- GRPO tự viết (dự phòng)
def run_manual(model, processor, data, out, entries, log):
    """GRPO với beta=0 và một lần cập nhật/lô sinh (giống mặc định TRL): ratio=1 nên
    loss = -mean_i[A_i * mean_t log pi(o_i,t)], A_i = (r_i - mean_nhóm) / (std_nhóm + 1e-4)."""
    G, P = CFG["num_generations"], CFG["prompts_per_step"]
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    opt = torch.optim.AdamW(params, lr=CFG["learning_rate"], weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    stats = HackStats(out / "grpo_hack_stats.jsonl")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    tok = processor.tokenizer
    it = 0
    for step in range(1, CFG["max_steps"] + 1):
        opt.zero_grad(set_to_none=True)
        step_rewards, step_stats = [], []
        for _ in range(P):
            e = entries[it % len(entries)]
            it += 1
            conv = [{"role": "user", "content": [
                {"type": "image", "image": Image.open(Path(data) / e["image_path"]).convert("RGB")},
                {"type": "text", "text": get_prompt()}]}]
            inp = processor.apply_chat_template(conv, tokenize=True, add_generation_prompt=True,
                                                return_dict=True, return_tensors="pt").to(model.device)
            n_prompt = inp["input_ids"].shape[1]
            model.eval()
            model.config.use_cache = True
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                gen = model.generate(**inp, do_sample=True, temperature=CFG["temperature"], top_p=1.0,
                                     top_k=0, max_new_tokens=CFG["max_completion_length"],
                                     num_return_sequences=G)
            comp_ids = gen[:, n_prompt:]
            texts = tok.batch_decode(comp_ids, skip_special_tokens=True)
            gt = gt_from_objects(e["objects"])
            rewards = torch.tensor([compute_reward(t, gt) for t in texts])
            step_rewards += rewards.tolist()
            step_stats.append(stats.add(texts, rewards.tolist(), [gt] * G))
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
            if rewards.std() < 1e-6:
                continue  # cả nhóm cùng reward: không có tín hiệu
            # mask completion: bỏ padding sau eos
            eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.pad_token_id, tok.eos_token_id}
            cmask = torch.ones_like(comp_ids, dtype=torch.float32)
            for i in range(G):
                ends = [j for j, t in enumerate(comp_ids[i].tolist()) if t in eos]
                if ends:
                    cmask[i, ends[0] + 1:] = 0
            model.train()
            model.config.use_cache = False
            full = gen
            attn = torch.ones_like(full)
            attn[:, n_prompt:] = (cmask > 0).long()
            rep = {k: v for k, v in inp.items() if k not in ("input_ids", "attention_mask")}
            rep = {k: (v.repeat(G, *([1] * (v.dim() - 1))) if k not in ("pixel_values", "image_grid_thw")
                       else torch.cat([v] * G, 0)) for k, v in rep.items()}
            if "mm_token_type_ids" in rep:
                pad = torch.zeros(G, full.shape[1] - rep["mm_token_type_ids"].shape[1],
                                  dtype=rep["mm_token_type_ids"].dtype, device=full.device)
                rep["mm_token_type_ids"] = torch.cat([rep["mm_token_type_ids"], pad], 1)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(input_ids=full, attention_mask=attn, **rep).logits
            logits = logits[:, n_prompt - 1:-1].float()
            logp = torch.log_softmax(logits, -1).gather(-1, comp_ids.unsqueeze(-1)).squeeze(-1)
            tok_mean = (logp * cmask.to(logp.device)).sum(1) / cmask.sum(1).clamp(min=1).to(logp.device)
            loss = -(adv.to(logp.device) * tok_mean).mean() / P
            scaler.scale(loss).backward()
            del logits, logp
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        scaler.step(opt)
        scaler.update()
        agg = {k: sum(s[k] for s in step_stats) / len(step_stats) for k in step_stats[0]}
        log({"kind": "step", "step": step, "grad_norm": round(gnorm, 4),
             "vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), **agg})
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="out")
    ap.add_argument("--name", default="e5")
    ap.add_argument("--init-adapter")
    ap.add_argument("--init-projector")
    ap.add_argument("--impl", default="trl", choices=["trl", "manual"])
    ap.add_argument("--max-steps", type=int, default=CFG["max_steps"])
    ap.add_argument("--infer-bs", type=int, default=8)
    a = ap.parse_args()
    CFG["max_steps"] = a.max_steps
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    adapter = _find(a.init_adapter, "adapter/adapter_config.json")
    projector = _find(a.init_projector, "projector.pt") if a.init_projector != "none" else None
    print("init adapter:", adapter, "| projector:", projector, flush=True)
    assert adapter, "không tìm thấy adapter SFT"

    log_path = out / "grpo_log.jsonl"
    log_path.unlink(missing_ok=True)

    def log(row):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row)[:400], flush=True)

    model, processor = load_trained(adapter, projector, trainable=True)
    for n, p in model.named_parameters():
        p.requires_grad = "lora_" in n and "visual" not in n
    model.enable_input_require_grads()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("trainable (LoRA LLM):", trainable, flush=True)

    entries = _entries(a.data)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    fn = run_trl if a.impl == "trl" else run_manual
    model = fn(model, processor, a.data, out, entries, log)
    train_min = (time.time() - t0) / 60
    peak = torch.cuda.max_memory_allocated() / 2**30
    model.save_pretrained(out / "adapter")
    run = {"name": a.name, "impl": a.impl, "init_adapter": adapter, "init_projector": projector,
           "trainable_params": trainable, "train_minutes": round(train_min, 2),
           "peak_vram_gb": round(peak, 2), "n_prompt_pool": len(entries), **CFG}
    json.dump(run, open(out / "run.json", "w"), indent=2)
    print("run:", run, flush=True)

    test_e = read_jsonl(Path(a.data) / "splits" / "test.jsonl")
    if a.max_steps < 10:  # chạy thử: chỉ suy luận vài ảnh
        test_e = test_e[:8]
    model.config.use_cache = True
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    _, mins = predict(model, processor, a.data, test_e, out / f"{a.name}.jsonl", bs=a.infer_bs)
    run["infer_minutes"] = round(mins, 2)
    json.dump(run, open(out / "run.json", "w"), indent=2)


if __name__ == "__main__":
    main()
