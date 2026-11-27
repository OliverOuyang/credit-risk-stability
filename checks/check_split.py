"""Time split: gaps between segments, calendar censoring, bad-count floor, and (with data) the frame."""
import re
from pathlib import Path

import pandas as pd

from checks.common import ROOT, load_frame, load_yaml, loan_quarter, main, quarter_months, yyyymm

ORDER = ["train", "valid", "test"]


def vintage_bads(root):
    """{vintage year: bads_main} from docs/bad_counts_raw.md (make bad-counts)."""
    p = Path(root) / "docs" / "bad_counts_raw.md"
    if not p.exists():
        return {}
    rows = [[c.strip() for c in line.strip("|").split("|")] for line in p.read_text().splitlines()
            if re.match(r"\|\s*\d{4}\s*\|", line)]
    head = [c.strip() for c in p.read_text().splitlines()[0].strip("|").split("|")]
    i = head.index("bads_main")
    return {int(r[0]): int(r[i]) for r in rows}


def run(root=ROOT, frame=None):
    split, target = load_yaml(root, "split"), load_yaml(root, "target")
    gap, segs = split["gap_months"], split["segments"]
    cutoff = yyyymm(target["censoring"]["performance_cutoff"])
    months = target["performance_months"]
    floor = split["min_bads_per_eval_segment"]
    v = []

    for a, b in zip(ORDER, ORDER[1:]):
        if quarter_months(segs[a]["end"])[1] + gap > quarter_months(segs[b]["start"])[0]:
            v.append(f"split.yaml: {a} ends {segs[a]['end']}, {b} starts {segs[b]['start']}: "
                     f"less than the {gap}-month gap")
    if quarter_months(segs["test"]["end"])[1] + gap > cutoff:
        v.append(f"split.yaml: test ends {segs['test']['end']} + {gap} months is after the "
                 f"performance cutoff {target['censoring']['performance_cutoff']}")
    if target["censoring"].get("rule") != "calendar":
        v.append("target.yaml: censoring.rule must be calendar (not per-loan observed months)")

    test_lo, test_hi = quarter_months(segs["test"]["start"])[0], quarter_months(segs["test"]["end"])[1]
    prev_hi = None
    for e in split.get("eval_segments", []):
        lo, hi = quarter_months(e["start"])[0], quarter_months(e["end"])[1]
        if lo < test_lo or hi > test_hi:
            v.append(f"split.yaml: eval segment {e['name']} is outside the test segment")
        if prev_hi is not None and lo <= prev_hi:
            v.append(f"split.yaml: eval segment {e['name']} overlaps the previous one")
        prev_hi = hi

    # Floor on bads per segment, from the per-vintage table.
    # shortcut: whole vintage years are summed; segments starting/ending mid-year overcount until quarterly counts exist.
    bads = vintage_bads(root)
    named = [(k, segs[k]) for k in ("train", "valid")] + [(e["name"], e) for e in split.get("eval_segments", [])]
    if bads:
        for name, s in named:
            n = sum(bads.get(y, 0) for y in range(int(s["start"][:4]), int(s["end"][:4]) + 1))
            if n < floor:
                v.append(f"split.yaml: segment {name} has {n} bads (docs/bad_counts_raw.md) < floor {floor}")

    frame = load_frame(root) if frame is None else frame
    if frame is None:
        return v
    multi = frame.groupby("loan_id")["segment"].nunique()
    for lid in multi[multi > 1].index:
        v.append(f"model frame: loan {lid} is in more than one segment")
    q = frame["loan_id"].map(loan_quarter)
    qm = q.map(lambda x: quarter_months(x)[0])
    expected = pd.Series(None, index=frame.index, dtype=object)
    for k in ORDER:
        lo, hi = quarter_months(segs[k]["start"])[0], quarter_months(segs[k]["end"])[1]
        expected[(qm >= lo) & (qm <= hi)] = k
    for i in frame.index[expected != frame["segment"]]:
        v.append(f"model frame: loan {frame.at[i, 'loan_id']} ({q[i]}) is in {frame.at[i, 'segment']}, "
                 f"split.yaml puts it in {expected[i]}")
    late = frame["first_payment_date"].map(yyyymm) + months - 1 > cutoff
    if late.any():
        v.append(f"model frame: {int(late.sum())} loans not censored (first payment + {months - 1} months "
                 f"after cutoff), e.g. {frame.loc[late, 'loan_id'].iloc[0]}")
    for name, s in named:
        lo, hi = quarter_months(s["start"])[0], quarter_months(s["end"])[1]
        n = int(frame.loc[(qm >= lo) & (qm <= hi), "bad"].sum())
        if n < floor:
            v.append(f"model frame: segment {name} has {n} bads < floor {floor}")
    return v


if __name__ == "__main__":
    main("split", run())
