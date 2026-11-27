"""Build data/interim/model_frame.parquet: one row per labelled loan in train/valid/test,
use:true origination features plus the meta columns in checks/common.py FRAME_META."""
from pathlib import Path

import pandas as pd

from crs.target import ORIG_COLS, PERF_COLS, ROOT, label, load_cfg, read_pipe

# Freddie Mac "not available" codes for numeric fields (User Guide, origination file).
MISSING = {"classic_fico": 9999, "mi_pct": 999, "number_of_units": 99, "original_cltv": 999,
           "original_dti": 999, "original_ltv": 999, "number_of_borrowers": 99}
NUMERIC = set(MISSING) | {"original_upb", "original_interest_rate", "original_loan_term"}


def quarter_index(q):
    return int(q[:4]) * 4 + int(q[5]) - 1


def segment_of(loan_id, segs):
    y = 2000 + loan_id.str[1:3].astype(int)
    qi = (y - 100 * (y > 2090)) * 4 + loan_id.str[4].astype(int) - 1
    seg = pd.Series(None, index=loan_id.index, dtype=object)
    for k, s in segs.items():
        seg[(qi >= quarter_index(s["start"])) & (qi <= quarter_index(s["end"]))] = k
    return seg


def build(raw_dir):
    feats = load_cfg("features")["origination"]
    cols = {i: f["name"] for i, f in enumerate(feats)}  # origination file order = features.yaml order
    use = [f["name"] for f in feats if f.get("use")]
    segs = load_cfg("split")["segments"]
    years = range(int(segs["train"]["start"][:4]), int(segs["test"]["end"][:4]) + 1)
    target = load_cfg("target")
    parts = []
    for y in years:
        d = Path(raw_dir) / f"sample_{y}"
        orig = read_pipe(d / f"sample_orig_{y}.txt", cols)
        orig["segment"] = segment_of(orig["loan_id"], segs)
        orig = orig[orig["segment"].notna()]
        if orig.empty:
            continue
        lab = label(orig[list(ORIG_COLS.values())], read_pipe(d / f"sample_perf_{y}.txt", PERF_COLS), target)
        parts.append(orig[["loan_id", "segment", "first_payment_date"] + use].merge(lab, on="loan_id"))
    df = pd.concat(parts, ignore_index=True)
    df = df[df["excluded"] == 0].reset_index(drop=True)
    for c in use:
        if c in NUMERIC:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            if c in MISSING:
                df.loc[df[c] == MISSING[c], c] = float("nan")
        else:
            df[c] = df[c].fillna("_blank")
    return df


if __name__ == "__main__":
    out = ROOT / "data" / "interim" / "model_frame.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    f = build(ROOT / "data" / "raw")
    f.to_parquet(out, index=False)
    print(f.groupby("segment").agg(loans=("bad", "size"), bads=("bad", "sum"), assisted=("assisted", "sum")))
