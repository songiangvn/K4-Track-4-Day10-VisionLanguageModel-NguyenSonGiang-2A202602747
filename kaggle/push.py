"""Đóng gói lab/ + scripts/ vào một kernel Kaggle (GPU T4) rồi push để chạy.

Chạy trên máy local (cần ~/.kaggle/kaggle.json):
  python kaggle/push.py smoke
  python kaggle/push.py e0
  python kaggle/push.py e1 -- --setup E1
  python kaggle/push.py e4_data_0.1 -- --setup E1 --train-frac 0.1
Sau đó:  kaggle kernels status <user>/lab10-<name>
         kaggle kernels output <user>/lab10-<name> -p runs/<name>
"""
import argparse
import base64
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = "lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes"
PIP = "transformers==5.19.0 peft==0.21.2 trl==1.9.2 accelerate==1.15.0 bitsandbytes==0.50.2"

KERNEL = r'''# Tự sinh bởi kaggle/push.py — đừng sửa tay.
import base64, io, os, subprocess, sys, tarfile
PAYLOAD = "{payload}"
SRC, DATA, OUT = "/tmp/src", "/tmp/data", "/kaggle/working/out"
tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD))).extractall(SRC)
os.makedirs(OUT, exist_ok=True)

def sh(cmd):
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, cwd=SRC)
    if r.returncode:
        raise SystemExit(f"FAILED ({{r.returncode}}): {{cmd}}")

sh("nvidia-smi -L")
sh("pip install -q {pip}")
sh("pip uninstall -y -q torchao")  # torchao 0.10 của Kaggle làm PEFT lỗi khi gắn LoRA vào nn.Linear (ViT, E3)
sh("python -c 'import torch, transformers, peft, trl, bitsandbytes as b; "
   "print(torch.__version__, transformers.__version__, peft.__version__, trl.__version__, b.__version__)'")
sh(f"python scripts/prepare_data.py --source /kaggle/input --out {{DATA}} --work /tmp/work")
sh(f"PYTHONUNBUFFERED=1 python -m {module} {cmd} --data {{DATA}} --out {{OUT}} {args}")
'''


def payload():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for sub in ("lab", "scripts"):
            for p in sorted((ROOT / sub).rglob("*.py")):
                if "__pycache__" not in p.parts:
                    tar.add(p, arcname=str(p.relative_to(ROOT)).replace(os.sep, "/"))
    return base64.b64encode(buf.getvalue()).decode()


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:  # mọi thứ sau '--' chuyển nguyên cho lab.run
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="smoke | e0 | e1 | e2 | e3 | e4_... | ...")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--from", dest="src", help="slug kernel SFT (vd lab10-e2) gắn làm input cho E5")
    a = ap.parse_args(argv)
    if a.name.startswith("e5"):
        module, cmd = "lab.grpo", ""
        args = " ".join(extra + ["--name", a.name])
    else:
        module = "lab.run"
        cmd = a.name if a.name in ("smoke", "e0") else "train"
        args = " ".join(extra + ([] if cmd != "train" else ["--name", a.name]))

    user = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["username"]
    slug = "lab10-" + a.name.replace("_", "-").replace(".", "")
    out = ROOT / "kaggle_build" / a.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "kernel.py").write_text(
        KERNEL.format(payload=payload(), pip=PIP, module=module, cmd=cmd, args=args), encoding="utf-8")
    meta = {
        "id": f"{user}/{slug}", "title": slug, "code_file": "kernel.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [DATASET], "competition_sources": [], "kernel_sources": [f"{user}/{a.src}"] if a.src else [],
        "model_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"kernel {meta['id']}: {module} {cmd} {args} | sources={meta['kernel_sources']}")
    if not a.dry:
        subprocess.run(["kaggle", "kernels", "push", "-p", str(out),
                        "--accelerator", "NvidiaTeslaT4"], check=True)


if __name__ == "__main__":
    main()
