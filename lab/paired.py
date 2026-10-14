"""Bootstrap ghép cặp (cùng mẫu ảnh cho cả hai setup) cho hiệu F1(B) - F1(A) trên test.

  python -m lab.paired runs/e2/out/e2.jsonl runs/e3/out/e3.jsonl
"""
import random
import sys

from scripts.labcommon import read_jsonl
from scripts.score import _gt_boxes, _match, _prf, _to_norm1000, parse_output


def rows(pred_path, entries, coord="norm1000"):
    preds = {r["image_id"]: r["raw"] for r in read_jsonl(pred_path)}
    out = []
    for e in entries:
        p = parse_output(preds.get(e["image_id"])) if e["image_id"] in preds else None
        p = _to_norm1000(p, coord, e["width"], e["height"]) if p else []
        per, _ = _match(_gt_boxes(e), p)
        out.append(tuple(sum(v[k] for v in per.values()) for k in range(3)))
    return out


def f1(rs, idx):
    tp = sum(rs[i][0] for i in idx)
    fp = sum(rs[i][1] for i in idx)
    fn = sum(rs[i][2] for i in idx)
    return _prf(tp, fp, fn)["f1"]


def main():
    a, b = sys.argv[1], sys.argv[2]
    entries = read_jsonl("data/splits/test.jsonl")
    ra, rb = rows(a, entries), rows(b, entries)
    n, rng, diffs = len(entries), random.Random(0), []
    for _ in range(2000):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(f1(rb, idx) - f1(ra, idx))
    diffs.sort()
    full = list(range(n))
    print(f"F1 A={f1(ra, full):.3f} B={f1(rb, full):.3f} diff={f1(rb, full) - f1(ra, full):+.3f} "
          f"CI95=[{diffs[50]:+.3f}, {diffs[1949]:+.3f}] P(diff<=0)={sum(d <= 0 for d in diffs) / 2000:.3f}")


if __name__ == "__main__":
    main()
