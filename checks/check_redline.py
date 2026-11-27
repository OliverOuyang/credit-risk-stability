"""Data and privacy red line: no raw/row-level loan data, notebook outputs, loan ids, secrets or
blacklisted keywords in anything committed. Runs on staged files (pre-commit) or all repo files."""
from fnmatch import fnmatch
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pandas as pd

from checks.common import ROOT, load_yaml, main

DATA_EXT = {".csv", ".tsv", ".parquet", ".feather", ".pkl", ".pickle", ".xlsx", ".xls", ".zip", ".gz", ".7z"}
LOAN_FILE = re.compile(r"sample_(orig|perf|svcg)_|historical_data_")
DATA_DIR_OK = {"data/manifest.sha256", "data/.gitkeep"}
SECRETS = [re.compile(p) for p in (
    r"AKIA[0-9A-Z]{16}", r"gh[pousr]_[A-Za-z0-9]{36}", r"github_pat_[A-Za-z0-9_]{40,}",
    r"sk-(ant-)?[A-Za-z0-9_-]{32,}", r"xox[baprs]-[A-Za-z0-9-]{10,}", r"AIza[0-9A-Za-z_-]{35}",
    r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----")]
RAW_ROW = re.compile(r"^[^|\s]*(\|[^|]*){19,}$")  # >= 20 pipe-separated fields: Freddie raw layout
MAX_RAW_ROWS = 20


def blacklist():
    p = Path(os.environ.get("CRS_REDLINE_BLACKLIST", Path.home() / ".config/crs/redline_blacklist.txt"))
    if not p.exists():
        print(f"[redline] note: keyword blacklist {p} not found, keyword check skipped", file=sys.stderr)
        return []
    return [w.strip().lower() for w in p.read_text().splitlines() if w.strip() and not w.startswith("#")]


def loan_re(root):
    pat = load_yaml(root, "features")["loan_id_regex"].strip("^$")
    return re.compile(rf"(?<![A-Za-z0-9]){pat}(?![0-9])")


def path_violations(f, allow):
    if f.startswith(("data/raw", "data/interim")):
        return [f"{f}: raw/interim data must not be committed"]
    if f.startswith("data/") and f not in DATA_DIR_OK:
        return [f"{f}: only data/manifest.sha256 may be committed under data/"]
    if LOAN_FILE.search(Path(f).name):
        return [f"{f}: Freddie Mac loan-level file"]
    if Path(f).suffix.lower() in DATA_EXT and not any(fnmatch(f, g) for g in allow):
        return [f"{f}: data file outside redline_allowlist.yaml allow_paths"]
    return []


def agg_violations(f, text, cfg, loan):
    df = pd.read_csv(io.StringIO(text), dtype=str)
    col, floor = cfg["count_column"], cfg["min_count"]
    v = [f"{f}: aggregate table has loan-id column {c}" for c in df.columns if "loan" in c.lower()]
    if any(loan.search(x) for x in df.astype(str).to_numpy().ravel()):
        v.append(f"{f}: aggregate table contains a loan id")
    if col not in df:
        return v + [f"{f}: aggregate table needs count column {col}"]
    small = pd.to_numeric(df[col], errors="coerce").fillna(0) < floor
    if small.any():
        v.append(f"{f}: {int(small.sum())} rows with {col} < {floor} (row {int(small.idxmax()) + 2})")
    return v


def content_violations(f, text, loan, words):
    v = []
    if f.endswith(".ipynb"):
        cells = json.loads(text).get("cells", [])
        if any(c.get("outputs") for c in cells):
            v.append(f"{f}: notebook has outputs; clear them before committing")
    raw_rows = 0
    for no, line in enumerate(text.splitlines(), 1):
        if loan.search(line):
            v.append(f"{f}:{no}: loan id")
        if any(p.search(line) for p in SECRETS):
            v.append(f"{f}:{no}: secret-like string")
        low = line.lower()
        v += [f"{f}:{no}: blacklisted keyword" for w in words if w in low][:1]
        raw_rows += bool(RAW_ROW.match(line))
    if raw_rows > MAX_RAW_ROWS:
        v.append(f"{f}: {raw_rows} raw loan-level rows (> {MAX_RAW_ROWS})")
    return v


def run(root, files, read=None):
    root = Path(root)
    read = read or (lambda f: (root / f).read_bytes())
    cfg = load_yaml(root, "redline_allowlist")
    loan, words = loan_re(root), blacklist()
    v = []
    for f in files:
        bad = path_violations(f, cfg.get("allow_paths", []))
        if bad:
            v += bad
            continue
        try:
            text = read(f).decode("utf-8")
        except UnicodeDecodeError:
            continue  # binary (png, ...): path rules only
        v += content_violations(f, text, loan, words)
        if fnmatch(f, cfg["agg"]["glob"]):
            v += agg_violations(f, text, cfg["agg"], loan)
    return v


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, capture_output=True, check=True).stdout


def repo_files(root):
    out = git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z").decode()
    return [f for f in out.split("\0") if f and (Path(root) / f).is_file()]


if __name__ == "__main__":
    if "--staged" in sys.argv:
        staged = git(ROOT, "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z").decode()
        files = [f for f in staged.split("\0") if f]
        main("redline", run(ROOT, files, read=lambda f: git(ROOT, "show", f":{f}")))
    main("redline", run(ROOT, repo_files(ROOT)))
