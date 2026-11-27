"""Leakage: model features must be known at origination; flag suspiciously strong single features."""
import pandas as pd

from checks.common import FRAME_META, ROOT, load_frame, load_yaml, main

ALLOW_KEYS = {"feature", "confirmed_by", "reason", "date"}


def single_auc(x, y):
    """AUC of one feature (categoricals encoded by their bad rate), direction-free."""
    if not pd.api.types.is_numeric_dtype(x):
        x = x.map(y.groupby(x).mean())
    ok = x.notna()
    x, y = x[ok].astype(float), y[ok].astype(int)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    auc = (x.rank()[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return max(auc, 1 - auc)


def run(root=ROOT, frame=None):
    feats = load_yaml(root, "features")
    allow_cfg = load_yaml(root, "leakage_allowlist")
    v = []
    perf = set(feats.get("performance", {}).get("used_for_target", []))
    model = []
    for f in feats.get("origination", []):
        if not f.get("use"):
            continue
        model.append(f["name"])
        if f.get("available_at") != "origination":
            v.append(f"features.yaml: {f['name']} is a model input but available_at={f.get('available_at')}")
        if f["name"] in perf:
            v.append(f"features.yaml: {f['name']} is a performance-file field used as a model input")

    allowed = set()
    for e in allow_cfg.get("allow") or []:
        if not ALLOW_KEYS <= set(e):
            v.append(f"leakage_allowlist.yaml: entry {e} missing {sorted(ALLOW_KEYS - set(e))}")
        else:
            allowed.add(e["feature"])

    frame = load_frame(root) if frame is None else frame
    if frame is None:
        print("[leakage] note: model frame absent, column and single-feature AUC checks skipped")
        return v
    for c in sorted(set(frame.columns) - FRAME_META - set(model)):
        v.append(f"model frame: column {c} is not a use:true origination feature in features.yaml")
    fit = frame[frame["segment"] == "train"] if "segment" in frame else frame
    threshold = allow_cfg.get("auc_threshold", 0.85)
    aucs = {c: single_auc(fit[c], fit["bad"]) for c in model if c in fit}
    for c, auc in aucs.items():
        if auc > threshold and c not in allowed:
            v.append(f"model frame: {c} single-feature AUC {auc:.3f} > {threshold}; "
                     f"confirm and add to leakage_allowlist.yaml, or drop it")
    top = max(aucs, key=aucs.get, default=None)
    print(f"[leakage] model frame checked: {len(frame)} rows, {frame.shape[1]} cols, "
          f"max single-feature AUC {top} {aucs.get(top, float('nan')):.3f}")
    return v


if __name__ == "__main__":
    main("leakage", run())
