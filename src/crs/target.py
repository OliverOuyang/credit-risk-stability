"""24-month default label and per-vintage bad counts, driven by config/target.yaml."""
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
# Positions from file_layout.xlsx (0-based): loan_id, period, delinquency status, loan age, ZBC,
# payment deferral, disaster flag, assistance plan.
PERF_COLS = {0: "loan_id", 1: "period", 3: "dlq", 4: "loan_age", 8: "zbc",
             24: "deferral", 28: "disaster", 29: "assistance"}
ORIG_COLS = {1: "first_payment_date", 19: "loan_id"}


def load_cfg(name):
    return yaml.safe_load((ROOT / "config" / f"{name}.yaml").read_text())


def read_pipe(path, cols):
    df = pd.read_csv(path, sep="|", header=None, usecols=list(cols), dtype=str)
    return df.rename(columns=cols)


def month_index(yyyymm):
    s = pd.Series(yyyymm).astype(str)
    return s.str[:4].astype(int) * 12 + s.str[4:6].astype(int) - 1


def label(orig, perf, cfg):
    """One row per loan: bad (0/1), assisted flag, excluded flag; censored loans dropped."""
    months = cfg["performance_months"]
    cutoff = int(cfg["censoring"]["performance_cutoff"].replace("-", ""))
    orig = orig.copy()
    keep = month_index(orig["first_payment_date"]).values + months - 1 <= month_index([cutoff])[0]
    orig = orig[keep]

    p = perf[perf["loan_id"].isin(orig["loan_id"])].copy()
    p["loan_age"] = pd.to_numeric(p["loan_age"], errors="coerce")
    p = p[p["loan_age"].between(1, months)]
    rule = cfg["bad_if_any_in_window"]
    dlq_num = pd.to_numeric(p["dlq"], errors="coerce")
    p["bad"] = ((dlq_num >= rule["delinquency_status_min"])
                | p["dlq"].isin(rule["delinquency_status_codes"])
                | p["zbc"].isin(rule["zero_balance_codes"]))
    ex = cfg["exclude"]
    p["excluded"] = p["zbc"].isin(ex["zero_balance_codes"]) | p["dlq"].isin(ex["delinquency_status_codes"])
    flag = cfg["assistance"]["flag_if_any_in_window"]
    p["assisted"] = (p["assistance"].isin(flag["borrower_assistance_plan"])
                     | p["disaster"].isin(flag["delinquency_due_to_disaster"])
                     | p["deferral"].isin(flag["payment_deferral_flag"]))
    agg = p.groupby("loan_id")[["bad", "excluded", "assisted"]].any().astype(int)
    out = orig[["loan_id"]].merge(agg, on="loan_id", how="left").fillna(0)
    out[["bad", "excluded", "assisted"]] = out[["bad", "excluded", "assisted"]].astype(int)
    out["vintage_year"] = id_year(out["loan_id"]).values
    return out


def id_year(loan_id):
    """Loan Identifier PYYQnXXXXXXX -> origination year."""
    y = 2000 + loan_id.str[1:3].astype(int)
    return y - 100 * (y > 2090)  # shortcut: 2-digit year, valid 1991-2090


def bad_counts(raw_dir):
    """Per-vintage table from data/raw/sample_YYYY/ files (sample_orig_YYYY.txt, sample_perf_YYYY.txt).
    Vintage = origination year in the Loan Identifier, the same key the split and frames use
    (a file can hold a loan from the next year, e.g. one 2009 loan in sample_2008)."""
    cfg = load_cfg("target")
    labs, raw = [], []
    for orig_path in sorted(Path(raw_dir).rglob("sample_orig_*.txt")):
        perf_path = orig_path.with_name(orig_path.name.replace("_orig_", "_perf_"))
        orig = read_pipe(orig_path, ORIG_COLS)
        raw.append(orig["loan_id"])
        labs.append(label(orig, read_pipe(perf_path, PERF_COLS), cfg))
    lab, ids = pd.concat(labs), pd.concat(raw)
    rows = []
    for y in sorted(set(id_year(ids))):
        v = lab[lab.vintage_year == y]
        main = v[(v.excluded == 0) & (v.assisted == 0)]
        rows.append({"vintage": y, "loans_raw": int((id_year(ids) == y).sum()), "loans_uncensored": len(v),
                     "loans_main": len(main), "bads_main": int(main.bad.sum()),
                     "bads_incl_assisted": int(v[v.excluded == 0].bad.sum()), "assisted": int(v.assisted.sum())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(bad_counts(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data" / "raw").to_markdown(index=False))
