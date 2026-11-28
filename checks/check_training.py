"""Training process: early stopping fired inside its range; train - valid AUC gaps within config/model.yaml
training_checks. Reads the "training" block model.py writes into outputs/metrics.json (committed, so CI runs it)."""
import json
from pathlib import Path

from checks.common import ROOT, load_yaml, main


def run(root=ROOT):
    root = Path(root)
    p = root / "outputs" / "metrics.json"
    if not p.exists():
        print("[training] note: outputs/metrics.json absent, skipped")
        return []
    t = json.loads(p.read_text()).get("training")
    if t is None:
        return ["outputs/metrics.json: no training block (rerun make model)"]
    c = load_yaml(root, "model")["training_checks"]
    v = []
    it, cap = t["lightgbm_best_iteration"], t["lightgbm_max_rounds"]
    if it >= cap:
        v.append(f"lightgbm stopped at the round cap {cap}: early stopping never fired")
    if it < c["min_best_iteration"]:
        v.append(f"lightgbm best iteration {it} < {c['min_best_iteration']}: valid AUC peaked at once")
    for m, g in t["auc_gap_train_minus_valid"].items():
        if g > c["max_auc_gap"]:
            v.append(f"{m}: train - valid AUC {g} > {c['max_auc_gap']} (overfitting or a train/valid mismatch)")
        if -g > c["max_valid_above_train"]:
            v.append(f"{m}: valid AUC beats train by {-g} > {c['max_valid_above_train']} (check valid for leakage)")
    ex = t["auc_gap_excess_lightgbm_over_scorecard"]
    if ex > c["max_gap_excess_lightgbm"]:
        v.append(f"lightgbm gap exceeds the scorecard's by {ex} > {c['max_gap_excess_lightgbm']} (overfitting)")
    return v


if __name__ == "__main__":
    main("training", run())
