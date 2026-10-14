"""Điều phối: smoke test, E0 zero-shot, huấn luyện E1–E4 và suy luận trên test.

Ví dụ:
  python -m lab.run smoke --data data --out out
  python -m lab.run e0    --data data --out out
  python -m lab.run train --data data --out out --name e1 --setup E1
  python -m lab.run train --data data --out out --name e4_data_0.1 --setup E1 --train-frac 0.1
"""
import argparse
import gc
import json
import math
import random
import time
from pathlib import Path

import torch
from PIL import Image

from lab.labels import get_prompt, objects_to_answer
from lab.vlm import (apply_setup, collate, encode_example, load_base, load_trained,
                     matched_modules, param_report, projector_state)
from scripts.labcommon import read_jsonl, write_jsonl


def _img(data, e):
    return Image.open(Path(data) / e["image_path"]).convert("RGB")


def _log(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


# ---------------------------------------------------------------- suy luận
@torch.no_grad()
def predict(model, processor, data, entries, out_path, coord="norm1000", bs=4, max_new_tokens=256):
    model.eval()
    model.config.use_cache = True
    rows, t0 = [], time.time()
    for i in range(0, len(entries), bs):
        chunk = entries[i:i + bs]
        convs = [[{"role": "user", "content": [
            {"type": "image", "image": _img(data, e)},
            {"type": "text", "text": get_prompt(coord, e["width"], e["height"])}]}] for e in chunk]
        inputs = processor.apply_chat_template(
            convs, tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt", padding=True).to(model.device)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        raws = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        rows += [{"image_id": e["image_id"], "raw": r} for e, r in zip(chunk, raws)]
        if i == 0 or (i // bs) % 10 == 0:
            print(f"[predict] {i + len(chunk)}/{len(entries)} {time.time() - t0:.0f}s | {raws[0][:160]!r}",
                  flush=True)
    write_jsonl(out_path, rows)
    return rows, (time.time() - t0) / 60


# ---------------------------------------------------------------- huấn luyện
def train(model, processor, data, entries, out_dir, setup, coord="norm1000", lr=2e-4, bs=2,
          grad_accum=4, max_steps=300, seed=42, log_every=10, save_every=100):
    out_dir = Path(out_dir)
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:  # GradScaler yêu cầu tham số huấn luyện ở fp32
        if p.dtype != torch.float32:
            p.data = p.data.float()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    warm = max(1, int(0.05 * max_steps))

    def lr_lambda(s):
        if s < warm:
            return (s + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, max_steps - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    pad_id = processor.tokenizer.pad_token_id

    rng = random.Random(seed)
    order, pos = [], 0

    def next_batch():
        nonlocal order, pos
        items = []
        for _ in range(bs):
            if pos >= len(order):
                order = list(range(len(entries)))
                rng.shuffle(order)
                pos = 0
            e = entries[order[pos]]
            pos += 1
            ans = objects_to_answer(e["objects"], coord, e["width"], e["height"])
            items.append(encode_example(processor, _img(data, e),
                                        get_prompt(coord, e["width"], e["height"]), ans))
        return {k: v.to(model.device) for k, v in collate(items, pad_id).items()}

    torch.cuda.reset_peak_memory_stats()
    log_path = out_dir / "train_log.jsonl"
    log_path.unlink(missing_ok=True)
    t0, run_loss = time.time(), 0.0
    for step in range(1, max_steps + 1):
        step_loss = 0.0
        for _ in range(grad_accum):
            batch = next_batch()
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss / grad_accum
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss không hữu hạn ở bước {step}: {loss.item()}")
            scaler.scale(loss).backward()
            step_loss += loss.item()
            if step == 1 and not getattr(train, "_checked", False):
                train._checked = True
                bad = [(n, str(p.dtype), str(p.grad.dtype)) for n, p in model.named_parameters()
                       if p.grad is not None and p.grad.dtype != torch.float32]
                nograd = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
                print(f"[check] grad không fp32: {len(bad)} {bad[:5]} | requires_grad nhưng không có grad: "
                      f"{len(nograd)} {nograd[:5]}", flush=True)
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        run_loss += step_loss
        if step % log_every == 0 or step == 1:
            _log(log_path, {"step": step, "loss": round(step_loss, 4),
                            "avg_loss": round(run_loss / (log_every if step > 1 else 1), 4),
                            "lr": sched.get_last_lr()[0], "grad_norm": round(gnorm, 3),
                            "vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                            "min": round((time.time() - t0) / 60, 2)})
            run_loss = 0.0
        if step % save_every == 0 or step == max_steps:
            model.save_pretrained(out_dir / "adapter")
            if setup in ("E2", "E3"):  # projector KHÔNG nằm trong adapter: lưu riêng
                torch.save(projector_state(model), out_dir / "projector.pt")
    return {"train_minutes": round((time.time() - t0) / 60, 2),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}


# ---------------------------------------------------------------- lệnh
def cmd_smoke(a):
    """Kiểm tra nhanh: cây module, 220 token ảnh, mask loss, số module LoRA, 3 bước train, suy luận 4 ảnh."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    train_e = read_jsonl(Path(a.data) / "splits" / "train.jsonl")
    test_e = read_jsonl(Path(a.data) / "splits" / "test.jsonl")
    model, processor = load_base()
    names = [n for n, _ in model.named_modules()]
    print("số module:", len(names))
    for n in names:
        if any(s in n for s in ("visual.merger", "deepstack_merger_list.0", "visual.blocks.0.",
                                "language_model.layers.0.")) or n.count(".") <= 2:
            print("  ", n)
    print("param_report:", param_report(model))
    print("dtypes:", {str(p.dtype) for p in model.parameters()})
    for s in ("E1", "E3"):
        print(f"matched {s}:", len(matched_modules(model, s)))

    e = next(x for x in train_e if len(x["objects"]) >= 2)
    ans = objects_to_answer(e["objects"])
    ex = encode_example(processor, _img(a.data, e), get_prompt(), ans)
    thw = ex["image_grid_thw"][0].tolist()
    print("image_grid_thw:", thw, "-> token ảnh =", thw[0] * thw[1] * thw[2] // 4)
    img_tok = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    print("số <|image_pad|> trong input_ids:", int((ex["input_ids"] == img_tok).sum()))
    print("seq len:", ex["input_ids"].shape[0], "| keys:", list(ex))
    learned = ex["labels"][ex["labels"] != -100]
    print("labels != -100 decode:", repr(processor.tokenizer.decode(learned)))
    print("answer gốc            :", repr(ans))

    model, info = apply_setup(model, "E2")
    print("setup E2:", info)
    proj = [(n, p.numel(), str(p.dtype), p.requires_grad) for n, p in model.named_parameters()
            if "merger" in n]
    print("projector params:", len(proj), sum(x[1] for x in proj))
    for x in proj:
        print("  ", x)
    print("trainable dtypes:", {str(p.dtype) for p in model.parameters() if p.requires_grad})
    res = train(model, processor, a.data, train_e, out, "E2", max_steps=3, log_every=1, save_every=3)
    print("train smoke:", res)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model, processor = load_trained(out / "adapter", out / "projector.pt")
    predict(model, processor, a.data, test_e[:4], out / "smoke_pred.jsonl")
    for r in read_jsonl(out / "smoke_pred.jsonl"):
        print(r)


def cmd_e0(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    test_e = read_jsonl(Path(a.data) / "splits" / "test.jsonl")
    model, processor = load_base()
    print("param_report:", param_report(model))
    _, mins = predict(model, processor, a.data, test_e, out / "e0.jsonl", bs=a.infer_bs)
    json.dump({"name": "e0", "infer_minutes": round(mins, 2),
               "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
               "param_report": param_report(model)},
              open(out / "run.json", "w"), indent=2)


def cmd_train(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    train_e = read_jsonl(Path(a.data) / "splits" / "train.jsonl")
    test_e = read_jsonl(Path(a.data) / "splits" / "test.jsonl")
    if a.train_frac < 1.0:
        k = max(1, round(len(train_e) * a.train_frac))
        train_e = random.Random(a.seed).sample(train_e, k)
    torch.manual_seed(a.seed)
    model, processor = load_base()
    model, info = apply_setup(model, a.setup, r=a.rank, alpha=2 * a.rank)
    print(a.name, info, "| n_train =", len(train_e), flush=True)
    res = train(model, processor, a.data, train_e, out, a.setup, coord=a.coord, lr=a.lr,
                bs=a.bs, grad_accum=a.grad_accum, max_steps=a.max_steps, seed=a.seed)
    run = {"name": a.name, "setup": a.setup, "rank": a.rank, "alpha": 2 * a.rank,
           "train_frac": a.train_frac, "n_train": len(train_e), "coord": a.coord, "lr": a.lr,
           "batch_size": a.bs, "grad_accum": a.grad_accum, "max_steps": a.max_steps, **info, **res}
    json.dump(run, open(out / "run.json", "w"), indent=2)
    print("run:", run, flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    proj = out / "projector.pt" if a.setup in ("E2", "E3") else None
    model, processor = load_trained(out / "adapter", proj)
    _, mins = predict(model, processor, a.data, test_e, out / f"{a.name}.jsonl",
                      coord=a.coord, bs=a.infer_bs)
    run["infer_minutes"] = round(mins, 2)
    json.dump(run, open(out / "run.json", "w"), indent=2)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "e0", "train"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="out")
    ap.add_argument("--name", default="e1")
    ap.add_argument("--setup", default="E1", choices=["E1", "E2", "E3"])
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--train-frac", type=float, default=1.0)
    ap.add_argument("--coord", default="norm1000", choices=["norm1000", "pixel"])
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--infer-bs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    {"smoke": cmd_smoke, "e0": cmd_e0, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
