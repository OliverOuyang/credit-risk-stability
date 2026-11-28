"""Data health: hand-made loans through the label rules; frames vs the committed per-vintage table
(docs/bad_counts_raw.md); duplicate and malformed loan ids; missing rates and value ranges; raw row counts."""
import re
from pathlib import Path

import pandas as pd

from checks.common import ROOT, load_yaml, main
from crs.target import ORIG_COLS, id_year, label, read_pipe

FRAMES = ("data/interim/model_frame.parquet", "data/interim/stability_frame.parquet")


def ym(m):
    """Month index (year*12 + month-1) -> 'YYYYMM'."""
    return f"{m // 12}{m % 12 + 1:02d}"


def label_cases(cfg):
    """(loan, first_payment_date, perf rows [(age, dlq, zbc, assistance, disaster, deferral)], expected).
    expected None = censored (dropped); else (bad, excluded, assisted). Built from target.yaml's cutoff and
    window so each case sits on the boundary it tests; the expected values are what the definition means."""
    n = cfg["performance_months"]
    c = int(cfg["censoring"]["performance_cutoff"].replace("-", "")[:4]) * 12 \
        + int(cfg["censoring"]["performance_cutoff"].replace("-", "")[4:6]) - 1
    last_kept = ym(c - n + 1)                      # first payment + n-1 months == cutoff
    lid = lambda i: "F10Q1" + f"{i:07d}"           # built here so the file holds no loan id
    return [
        ("90+ dpd in the last window month", lid(1), "201003", [(n, "3", None, None, None, None)], (1, 0, 0)),
        ("90+ dpd one month after the window", lid(2), "201003", [(n + 1, "3", None, None, None, None)], (0, 0, 0)),
        ("60 dpd only", lid(3), "201003", [(6, "2", None, None, None, None)], (0, 0, 0)),
        ("prepaid early: good, kept", lid(4), "201003", [(3, "0", None, None, None, None),
                                                         (6, "0", "01", None, None, None)], (0, 0, 0)),
        ("prepaid after a 90+ spell: still bad", lid(5), "201003", [(4, "3", None, None, None, None),
                                                                    (8, "0", "01", None, None, None)], (1, 0, 0)),
        ("REO acquisition", lid(6), "201003", [(10, "RA", None, None, None, None)], (1, 0, 0)),
        ("short sale", lid(7), "201003", [(11, "0", "03", None, None, None)], (1, 0, 0)),
        ("forbearance in window", lid(8), "201003", [(8, "3", None, "F", None, None)], (1, 0, 1)),
        ("disaster flag in window", lid(9), "201003", [(8, "0", None, None, "Y", None)], (0, 0, 1)),
        ("forbearance after the window: not assisted", lid(10), "201003",
         [(n + 2, "0", None, "F", None, None)], (0, 0, 0)),
        ("repayment plan / deferral: not assisted", lid(11), "201003",
         [(9, "1", None, "R", None, "Y")], (0, 0, 0)),
        ("defect before credit event: excluded", lid(12), "201003", [(5, "0", "96", None, None, None)], (0, 1, 0)),
        ("status not available: excluded", lid(13), "201003", [(5, "XX", None, None, None, None)], (0, 1, 0)),
        ("window ends exactly at cutoff: kept", lid(14), last_kept, [(1, "0", None, None, None, None)], (0, 0, 0)),
        ("window ends one month after cutoff: censored", lid(15), ym(c - n + 2),
         [(1, "0", None, None, None, None)], None),
    ]


def check_labels(cfg):
    cases = label_cases(cfg)
    orig = pd.DataFrame({"loan_id": [x[1] for x in cases], "first_payment_date": [x[2] for x in cases]})
    perf = pd.DataFrame([dict(loan_id=x[1], period="", loan_age=str(a), dlq=d, zbc=z, assistance=s,
                              disaster=dis, deferral=dfr) for x in cases for a, d, z, s, dis, dfr in x[3]])
    out = label(orig, perf, cfg).set_index("loan_id")
    v = []
    for name, lid, _, _, want in cases:
        got = None if lid not in out.index else tuple(int(out.at[lid, k]) for k in ("bad", "excluded", "assisted"))
        if got != want:
            v.append(f"label rule '{name}': got (bad, excluded, assisted) {got}, expected {want}")
    return v


def committed_table(root):
    p = Path(root) / "docs" / "bad_counts_raw.md"
    lines = [ln for ln in p.read_text().splitlines() if ln.startswith("|")] if p.exists() else []
    if len(lines) < 3:
        return None
    split = lambda ln: [c.strip() for c in ln.strip("|").split("|")]
    return pd.DataFrame([split(ln) for ln in lines[2:]], columns=split(lines[0])).astype(int).set_index("vintage")


def check_frame(name, frame, table, cfg, features):
    v = []
    dup = frame["loan_id"][frame["loan_id"].duplicated()]
    if len(dup):
        v.append(f"{name}: {len(dup)} duplicate loan ids, e.g. {dup.iloc[0]}")
    bad_id = frame["loan_id"][~frame["loan_id"].str.match(features["loan_id_regex"])]
    if len(bad_id):
        v.append(f"{name}: {len(bad_id)} loan ids not matching {features['loan_id_regex']}")
    if table is not None:
        g = frame.assign(main=frame["assisted"] == 0, main_bad=(frame["assisted"] == 0) & (frame["bad"] == 1))
        got = g.groupby("vintage_year").agg(loans_main=("main", "sum"), bads_main=("main_bad", "sum"),
                                            bads_incl_assisted=("bad", "sum"))
        for y, r in got.iterrows():
            if y not in table.index:
                v.append(f"{name}: vintage {y} not in docs/bad_counts_raw.md")
                continue
            for k in got.columns:
                if int(r[k]) != table.at[y, k]:
                    v.append(f"{name}: vintage {y} {k} {int(r[k])}, docs/bad_counts_raw.md says {table.at[y, k]}")
    for c, (lo, hi) in cfg["ranges"].items():
        x = frame[c].dropna()
        out = x[(x < lo) | (x > hi)]
        if len(out):
            v.append(f"{name}: {c} has {len(out)} values outside [{lo}, {hi}], e.g. {out.iloc[0]}")
    off = frame["original_upb"].dropna() % cfg["upb_multiple"] != 0
    if off.any():
        v.append(f"{name}: original_upb has {int(off.sum())} values not a multiple of {cfg['upb_multiple']}")
    use = [f["name"] for f in features["origination"] if f.get("use")]
    for c in use:
        if c not in frame:
            v.append(f"{name}: feature {c} missing from the frame")
            continue
        miss = frame[c].isna() | (frame[c] == "_blank") if frame[c].dtype == object else frame[c].isna()
        limit = cfg["max_missing"].get(c, cfg["max_missing"]["default"])
        if miss.mean() > limit:
            v.append(f"{name}: {c} missing {miss.mean():.2%} > {limit:.0%}")
    return v


def check_raw(raw_dir, table):
    """Raw origination rows per vintage vs the table's loans_raw (catches a stale table or a changed download)."""
    files = sorted(Path(raw_dir).rglob("sample_orig_*.txt"))
    if not files or table is None:
        return []
    ids = pd.concat([read_pipe(f, {k: v for k, v in ORIG_COLS.items() if v == "loan_id"})["loan_id"] for f in files])
    got = id_year(ids).value_counts()
    return [f"raw files: vintage {y} has {got.get(y, 0)} origination rows, docs/bad_counts_raw.md says "
            f"{table.at[y, 'loans_raw']}" for y in table.index if got.get(y, 0) != table.at[y, "loans_raw"]]


def run(root=ROOT, frames=None, raw_dir=None):
    root = Path(root)
    cfg, target, features = load_yaml(root, "data_checks"), load_yaml(root, "target"), load_yaml(root, "features")
    table = committed_table(root)
    v = check_labels(target)
    if table is None:
        v.append("docs/bad_counts_raw.md missing or empty (make bad-counts)")
    if frames is None:
        frames = {f: pd.read_parquet(root / f) for f in FRAMES if (root / f).exists()}
    if not frames:
        print("[data] note: no frames in data/interim, frame checks skipped")
    for name, frame in frames.items():
        print(f"[data] {name}: {len(frame)} rows checked")
        v += check_frame(name, frame, table, cfg, features)
    raw = Path(raw_dir) if raw_dir else root / "data" / "raw"
    if raw.exists():
        v += check_raw(raw, table)
    else:
        print("[data] note: data/raw absent, raw row counts skipped")
    return v


if __name__ == "__main__":
    main("data", run())
