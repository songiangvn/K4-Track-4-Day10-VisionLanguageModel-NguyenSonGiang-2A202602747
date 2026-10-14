"""Tải kết quả nhẹ (log, *.jsonl, run.json) của một kernel về runs/<name>/ rồi chấm bằng scripts/score.py.

  python kaggle/fetch.py e1            # tải + chấm test -> report/evidence/e1.json
  python kaggle/fetch.py e4_data_0.1   # tương tự, evidence/e4_data_0.1.json
  python kaggle/fetch.py e1 --no-score
Không tải trọng số (adapter, projector.pt) để tiết kiệm băng thông.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--no-score", action="store_true")
    ap.add_argument("--coord", default="norm1000")
    a = ap.parse_args()
    user = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["username"]
    slug = "lab10-" + a.name.replace("_", "-").replace(".", "")
    dest = ROOT / "runs" / a.name
    dest.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUTF8": "1"}
    subprocess.run(["kaggle", "kernels", "output", f"{user}/{slug}", "-p", str(dest), "-o", "-q",
                    "--file-pattern", r".*\.(jsonl|json|log)$"], check=True, env=env)
    log = dest / f"{slug}.log"
    if log.exists():  # log Kaggle là JSON [{"data": ...}] -> văn bản thường
        try:
            text = "".join(r["data"] for r in json.loads(log.read_text(encoding="utf-8")))
            (dest / "kernel.log").write_text(text, encoding="utf-8")
            tail = [ln for ln in text.splitlines() if ln.strip() and "Loading weights" not in ln][-15:]
            print("\n".join(tail))
        except ValueError:
            pass
    run_p, pred_p = dest / "out" / "run.json", dest / "out" / f"{a.name}.jsonl"
    run = json.loads(run_p.read_text()) if run_p.exists() else {}
    print("run.json:", run)
    if a.no_score or not pred_p.exists():
        if not pred_p.exists():
            print("chưa có", pred_p)
        return
    cmd = [sys.executable, "scripts/score.py", "--pred", str(pred_p), "--name", a.name.upper(),
           "--out", f"report/evidence/{a.name}.json", "--coord", a.coord]
    for k in ("trainable_params", "peak_vram_gb", "train_minutes"):
        if k in run:
            cmd += ["--meta", f"{k}={run[k]}"]
    if "infer_minutes" in run:
        cmd += ["--meta", f"infer_minutes={run['infer_minutes']}"]
    subprocess.run(cmd, check=True, cwd=ROOT, env={**env, "PYTHONIOENCODING": "utf-8"})


if __name__ == "__main__":
    main()
