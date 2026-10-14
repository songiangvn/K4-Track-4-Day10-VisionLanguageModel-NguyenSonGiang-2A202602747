"""Ảnh minh hoạ (GT xanh lá, dự đoán đỏ) và bảng tổng hợp từ report/evidence/*.json.

  python -m lab.viz gt     --out report/figures/gt_samples.jpg
  python -m lab.viz preds  --pred runs/e1/out/e1.jsonl --tag e1 --n 6
  python -m lab.viz table
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from scripts.labcommon import read_jsonl
from scripts.score import _gt_boxes, _match, _to_norm1000, parse_output

COL = {"gt": (0, 220, 0), "pred": (255, 40, 40)}


def draw(entry, data="data", pred=None, title=""):
    im = Image.open(Path(data) / entry["image_path"]).convert("RGB")
    d = ImageDraw.Draw(im)
    W, H = im.size

    def box(bb, lab, col, up):
        x1, y1, x2, y2 = bb[0] * W / 1000, bb[1] * H / 1000, bb[2] * W / 1000, bb[3] * H / 1000
        d.rectangle([x1, y1, x2, y2], outline=col, width=2)
        d.text((x1 + 2, y1 - 11 if up else y2 + 1), lab, fill=col)

    for g in _gt_boxes(entry):
        box(g["bbox"], g["label"], COL["gt"], True)
    for p in pred or []:
        box(p["bbox"], p["label"], COL["pred"], False)
    if title:
        d.rectangle([0, 0, W, 14], fill=(0, 0, 0))
        d.text((3, 1), title, fill=(255, 255, 255))
    return im


def grid(images, cols=2):
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    g = Image.new("RGB", (w * cols, h * rows), "white")
    for i, im in enumerate(images):
        g.paste(im, ((i % cols) * w, (i // cols) * h))
    return g


def per_image(entries, preds, coord="norm1000"):
    rows = []
    for e in entries:
        raw = preds.get(e["image_id"])
        parsed = parse_output(raw) if raw is not None else None
        pred = _to_norm1000(parsed, coord, e["width"], e["height"]) if parsed else []
        per, _ = _match(_gt_boxes(e), pred)
        tp, fp, fn = (sum(v[k] for v in per.values()) for k in range(3))
        rows.append({"e": e, "pred": pred, "valid": parsed is not None, "tp": tp, "fp": fp, "fn": fn})
    return rows


def cmd_gt(a):
    entries = read_jsonl(Path(a.data) / "splits" / "train.jsonl")
    pick = [e for e in entries if len(e["objects"]) >= 2][:: max(1, len(entries) // 40)][:6]
    ims = [draw(e, a.data, title=f"{e['image_id']} | {e['session']} | GT={len(e['objects'])}")
           for e in pick]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    grid(ims).save(a.out, quality=90)
    print("saved", a.out)


def cmd_preds(a):
    entries = read_jsonl(Path(a.data) / "splits" / f"{a.split}.jsonl")
    preds = {r["image_id"]: r["raw"] for r in read_jsonl(a.pred)}
    rows = per_image(entries, preds, a.coord)
    good = sorted([r for r in rows if r["tp"] and not r["fp"] and not r["fn"]],
                  key=lambda r: -r["tp"])
    bad_fp = sorted([r for r in rows if r["fp"]], key=lambda r: -r["fp"])
    bad_fn = sorted([r for r in rows if r["fn"] and not r["fp"]], key=lambda r: -r["fn"])
    k = a.n // 3 or 1
    chosen = [("ĐÚNG", r) for r in good[:k]] + [("SAI-FP", r) for r in bad_fp[:k]] + \
             [("SAI-FN", r) for r in bad_fn[:a.n - 2 * k]]
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ims = []
    for kind, r in chosen:
        t = f"{a.tag.upper()} {kind} {r['e']['image_id'][-22:]} TP={r['tp']} FP={r['fp']} FN={r['fn']}"
        ims.append(draw(r["e"], a.data, r["pred"], t))
        print(t)
    path = out_dir / f"{a.tag}_examples.jpg"
    grid(ims).save(path, quality=90)
    print("saved", path)


def cmd_table(a):
    ev = {p.stem: json.loads(p.read_text(encoding="utf-8"))
          for p in sorted(Path(a.evidence).glob("e*.json"))}
    print("| Mã | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | valid | F1 | CI95 | FP/ảnh | "
          "F1 pothole | F1 crack | F1 manhole |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for k, d in ev.items():
        m, pc = d.get("meta", {}), d["per_class"]
        print(f"| {k} | {m.get('trainable_params', 0):,} | {m.get('peak_vram_gb', '')} | "
              f"{m.get('train_minutes', '')} | {d['parse_valid_rate']:.3f} | {d['micro']['f1']:.3f} | "
              f"{d['f1_ci95'][0]:.3f}–{d['f1_ci95'][1]:.3f} | {d['fp_per_image']:.2f} | "
              f"{pc['pothole']['f1']:.3f} | {pc['crack']['f1']:.3f} | {pc['manhole']['f1']:.3f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gt", "preds", "table"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--pred")
    ap.add_argument("--coord", default="norm1000")
    ap.add_argument("--tag", default="e1")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default="report/figures/gt_samples.jpg")
    ap.add_argument("--out-dir", default="report/figures")
    ap.add_argument("--evidence", default="report/evidence")
    a = ap.parse_args()
    {"gt": cmd_gt, "preds": cmd_preds, "table": cmd_table}[a.cmd](a)


if __name__ == "__main__":
    main()
