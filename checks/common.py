"""Shared helpers for checks/. Each check exposes run(root, ...) -> list of violation strings."""
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
# Contract for 2c: the modelling frame, one row per loan, features plus these meta columns.
FRAME = "data/interim/model_frame.parquet"
FRAME_META = {"loan_id", "segment", "bad", "first_payment_date", "vintage_year", "assisted", "excluded"}


def load_yaml(root, name):
    p = Path(root) / "config" / f"{name}.yaml"
    return (yaml.safe_load(p.read_text()) or {}) if p.exists() else {}


def load_frame(root):
    p = Path(root) / FRAME
    return pd.read_parquet(p) if p.exists() else None


def quarter_months(q):
    """'2010Q4' -> (first, last) month index (year*12 + month-1)."""
    y, n = int(q[:4]), int(q[5])
    first = y * 12 + (n - 1) * 3
    return first, first + 2


def yyyymm(s):
    s = str(s).replace("-", "")
    return int(s[:4]) * 12 + int(s[4:6]) - 1


def loan_quarter(loan_id):
    """Loan Identifier PYYQnXXXXXXX -> '20YYQn'."""
    y = 2000 + int(loan_id[1:3])
    y -= 100 if y > 2090 else 0  # shortcut: 2-digit year, valid 1991-2090
    return f"{y}Q{loan_id[4]}"


def main(name, violations):
    for v in violations:
        print(f"[{name}] {v}", file=sys.stderr)
    print(f"[{name}] {'FAIL ' + str(len(violations)) if violations else 'ok'}")
    sys.exit(1 if violations else 0)
