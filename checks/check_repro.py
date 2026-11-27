"""Reproducibility: locked deps; dev-subset metrics identical across two runs and to the committed reference.
Raw files are verified separately by `make verify-raw` (re-hash vs data/manifest.sha256)."""
import json
import subprocess
from pathlib import Path

from checks.common import ROOT, load_yaml, main


def flatten(d, prefix=""):
    if isinstance(d, dict):
        return {k: x for key, val in d.items() for k, x in flatten(val, f"{prefix}{key}.").items()}
    if isinstance(d, list):
        return {k: x for i, val in enumerate(d) for k, x in flatten(val, f"{prefix}{i}.").items()}
    return {prefix.rstrip("."): d}


def diff(a, b, tol, label):
    fa, fb = flatten(a), flatten(b)
    v = [f"{label}: key {k} only on one side" for k in sorted(set(fa) ^ set(fb))]
    for k in sorted(set(fa) & set(fb)):
        x, y = fa[k], fb[k]
        num = isinstance(x, (int, float)) and isinstance(y, (int, float))
        if (num and abs(x - y) > tol) or (not num and x != y):
            v.append(f"{label}: {k} {x} vs {y}")
    return v


def run(root=ROOT):
    root = Path(root)
    cfg = load_yaml(root, "repro")
    v = [] if (root / "uv.lock").exists() else ["uv.lock missing: dependency versions are not locked"]
    if "seed" not in cfg:
        v.append("repro.yaml: seed missing")
    if not cfg.get("dev_command"):
        return v
    out = root / cfg["dev_output"]
    runs = []
    for i in (1, 2):
        r = subprocess.run(cfg["dev_command"], shell=True, cwd=root, capture_output=True, text=True)
        if r.returncode != 0:
            return v + [f"dev_command run {i} failed: {r.stderr.strip()[-500:]}"]
        runs.append(json.loads(out.read_text()))
    tol = cfg.get("tolerance", 1e-6)
    v += diff(runs[0], runs[1], tol, "run 1 vs run 2")
    if cfg.get("reference"):
        v += diff(json.loads((root / cfg["reference"]).read_text()), runs[1], tol, "reference vs this run")
    return v


if __name__ == "__main__":
    main("repro", run())
