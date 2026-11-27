"""WOE scorecard and LightGBM on the same time split; AUC/KS with bootstrap CIs and calibration per
segment. Real run: model frame -> outputs/metrics.json + reports/figures. `--dev`: synthetic frame
-> outputs/dev_metrics.json (check_repro, runs without raw data)."""
import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from crs.target import ROOT, load_cfg
from crs.frame import NUMERIC

R = 6  # decimals kept in metrics files


def auc(y, s):
    r = pd.Series(s).rank().to_numpy()
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ks(y, s):
    o = np.argsort(-s, kind="stable")
    yo = y[o]
    tpr = np.cumsum(yo) / yo.sum()
    fpr = np.cumsum(1 - yo) / (len(yo) - yo.sum())
    return float(np.max(np.abs(tpr - fpr)))


def evaluate(y, p, reps, rng, bins, level):
    y, p = np.asarray(y, int), np.asarray(p, float)
    boot = {"auc": [], "ks": []}
    for _ in range(reps):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            boot["auc"].append(auc(y[i], p[i]))
            boot["ks"].append(ks(y[i], p[i]))
    out = {}
    for name, fn in (("auc", auc), ("ks", ks)):
        lo, hi = np.percentile(boot[name], [50 * (1 - level), 50 * (1 + level)])
        out[name] = {"est": round(fn(y, p), R), "lo": round(lo, R), "hi": round(hi, R)}
    q = pd.qcut(pd.Series(p).rank(method="first"), bins, labels=False)
    cal = pd.DataFrame({"q": q, "p": p, "y": y}).groupby("q").agg(n=("y", "size"), mean_pred=("p", "mean"),
                                                                    bad_rate=("y", "mean"))
    out["brier"] = round(float(np.mean((p - y) ** 2)), R)
    out["mean_pred"] = round(float(p.mean()), R)
    out["bad_rate"] = round(float(y.mean()), R)
    out["calibration"] = [{"bin": int(k), "n": int(r.n), "mean_pred": round(r.mean_pred, R),
                           "bad_rate": round(r.bad_rate, R)} for k, r in cal.iterrows()]
    return out


class Woe:
    """Per-feature bins fitted on train: numeric quantile edges + missing bin; categorical levels with
    rare ones pooled into _other. shortcut: bins are not forced monotonic; add if a reviewer needs it."""

    def __init__(self, cfg):
        self.cfg = cfg

    def _bin(self, c, x):
        if c in NUMERIC:
            b = np.searchsorted(self.edges[c], x.to_numpy(), side="right").astype(object)
            b[x.isna().to_numpy()] = "_missing"
            return pd.Series(b, index=x.index).astype(str)
        return x.where(x.isin(self.levels[c]), "_other").astype(str)

    def fit(self, X, y):
        self.edges, self.levels, self.table, self.iv = {}, {}, {}, {}
        sm = self.cfg["smoothing"]
        B, G = y.sum(), len(y) - y.sum()
        for c in X:
            x = X[c]
            if c in NUMERIC:
                qs = np.linspace(0, 1, self.cfg["max_bins"] + 1)[1:-1]
                self.edges[c] = np.unique(np.nanquantile(x.to_numpy(float), qs))
            else:
                share = x.value_counts(normalize=True)
                self.levels[c] = set(share[share >= self.cfg["min_bin_share"]].index)
            t = pd.DataFrame({"b": self._bin(c, x), "y": y}).groupby("b")["y"].agg(["sum", "size"])
            pb, pg = (t["sum"] + sm) / B, (t["size"] - t["sum"] + sm) / G
            woe = np.log(pb / pg)
            self.table[c] = woe.to_dict()
            self.iv[c] = float(((pb - pg) * woe).sum())
        return self

    def transform(self, X):
        return pd.DataFrame({c: self._bin(c, X[c]).map(self.table[c]).fillna(0.0) for c in X}, index=X.index)


def features():
    return [f["name"] for f in load_cfg("features")["origination"] if f.get("use")]


def fit_scorecard(train, cfg):
    feats = features()
    woe = Woe(cfg["woe"]).fit(train[feats], train["bad"])
    kept = [c for c in feats if woe.iv[c] >= cfg["woe"]["iv_min"]]
    lr = LogisticRegression(C=cfg["logistic"]["C"], max_iter=2000)
    lr.fit(woe.transform(train[kept]), train["bad"])
    info = {c: {"iv": round(woe.iv[c], R), "kept": c in kept} for c in feats}
    for c, w in zip(kept, lr.coef_[0]):
        info[c]["coef"] = round(float(w), R)
    predict = lambda d: lr.predict_proba(woe.transform(d[kept]))[:, 1]
    return predict, {"features": info, "intercept": round(float(lr.intercept_[0]), R),
                     "n_candidates": len(feats), "n_kept": len(kept)}


def fit_lightgbm(train, valid, cfg, seed):
    feats = features()
    cats = {c: pd.CategoricalDtype(sorted(train[c].unique())) for c in feats if c not in NUMERIC}
    # Levels unseen in train become missing (LightGBM's own handling of NaN categories).
    prep = lambda d: pd.DataFrame({c: d[c].where(d[c].isin(cats[c].categories)).astype(cats[c]) if c in cats
                                   else d[c] for c in feats})
    p = cfg["lightgbm"]
    params = {"objective": "binary", "metric": "auc", "learning_rate": p["learning_rate"],
              "num_leaves": p["num_leaves"], "min_child_samples": p["min_child_samples"],
              "feature_fraction": p["feature_fraction"], "lambda_l2": p["lambda_l2"],
              "cat_smooth": p["cat_smooth"], "min_data_per_group": p["min_data_per_group"],
              "num_threads": p["num_threads"], "seed": seed, "deterministic": True,
              "force_col_wise": True, "verbose": -1}
    dtrain = lgb.Dataset(prep(train), train["bad"])
    dvalid = lgb.Dataset(prep(valid), valid["bad"], reference=dtrain)
    booster = lgb.train(params, dtrain, p["max_rounds"], valid_sets=[dvalid],
                        callbacks=[lgb.early_stopping(p["early_stopping_rounds"], verbose=False)])
    gain = pd.Series(booster.feature_importance("gain"), index=feats)
    share = (gain / gain.sum()).sort_values(ascending=False)
    predict = lambda d: booster.predict(prep(d), num_iteration=booster.best_iteration)
    return predict, {"best_iteration": int(booster.best_iteration),
                     "gain_share": {c: round(float(v), R) for c, v in share.items()}}


def segments(frame):
    """Named evaluation slices: train, valid, each test eval segment, whole test."""
    q = frame["loan_id"].str[1:3].astype(int) * 4 + frame["loan_id"].str[4].astype(int) - 1
    out = {"train": frame["segment"] == "train", "valid": frame["segment"] == "valid"}
    for e in load_cfg("split")["eval_segments"]:
        lo = (int(e["start"][2:4])) * 4 + int(e["start"][5]) - 1
        hi = (int(e["end"][2:4])) * 4 + int(e["end"][5]) - 1
        out[e["name"]] = (frame["segment"] == "test") & q.between(lo, hi)
    out["test"] = frame["segment"] == "test"
    return out


def run(frame, cfg, seed, reps):
    main = frame[(frame["assisted"] == 0)].reset_index(drop=True)
    train, valid = main[main.segment == "train"], main[main.segment == "valid"]
    rng = np.random.default_rng(seed)
    sc_pred, sc_info = fit_scorecard(train, cfg)
    gb_pred, gb_info = fit_lightgbm(train, valid, cfg, seed)
    bins, level = cfg["evaluation"]["calibration_bins"], cfg["evaluation"]["ci_level"]
    sample, res = {}, {"scorecard": {**sc_info, "segments": {}}, "lightgbm": {**gb_info, "segments": {}}}
    for name, m in segments(main).items():
        d = main[m]
        sample[name] = {"loans": int(len(d)), "bads": int(d.bad.sum()), "bad_rate": round(float(d.bad.mean()), R)}
        for model, pred in (("scorecard", sc_pred), ("lightgbm", gb_pred)):
            res[model]["segments"][name] = evaluate(d["bad"], pred(d), reps, rng, bins, level)
    return {"sample": sample, **res}, {"scorecard": sc_pred, "lightgbm": gb_pred}


def dev_frame(cfg, seed):
    """Synthetic loans with the frame's columns; bad depends on a few features, base rate drifts."""
    rng = np.random.default_rng(seed)
    n = cfg["dev"]["loans_per_segment"]
    rows = []
    for seg, yy, shift in (("train", 7, 0.0), ("valid", 13, -0.8), ("test", 18, -1.2), ("test", 21, -1.0),
                           ("test", 23, -0.6)):
        d = pd.DataFrame({c: rng.normal(size=n) for c in sorted(NUMERIC)})
        for c in [c for c in features() if c not in NUMERIC]:
            d[c] = rng.choice(["A", "B", "C"], n)
        z = -3 + shift - 0.8 * d["classic_fico"] + 0.5 * d["original_dti"] + 0.4 * (d["channel"] == "B")
        d["bad"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
        d["segment"], d["assisted"], d["excluded"] = seg, 0, 0
        d["loan_id"] = [f"F{yy:02d}Q{1 + i % 4}{i:07d}" for i in range(n)]
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def calibration_figure(metrics, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [k for k in metrics["sample"] if k != "test"]
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.2), sharey=True)
    for ax, name in zip(axes, names):
        top = 0
        for model, color in (("scorecard", "#2563eb"), ("lightgbm", "#ea580c")):
            cal = metrics[model]["segments"][name]["calibration"]
            x, y = [c["mean_pred"] for c in cal], [c["bad_rate"] for c in cal]
            ax.plot(x, y, "o-", color=color, label=model, ms=3)
            top = max(top, *x, *y)
        ax.plot([0, top], [0, top], ":", color="grey")
        ax.set_title(name)
        ax.set_xlabel("mean predicted PD")
    axes[0].set_ylabel("observed bad rate")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main(argv):
    cfg = load_cfg("model")
    seed = load_cfg("repro")["seed"]
    (ROOT / "outputs").mkdir(exist_ok=True)
    if "--dev" in argv:
        m, _ = run(dev_frame(cfg, seed), cfg, seed, cfg["dev"]["bootstrap_reps"])
        (ROOT / "outputs" / "dev_metrics.json").write_text(json.dumps(m, indent=1, sort_keys=True))
        return
    frame = pd.read_parquet(ROOT / "data" / "interim" / "model_frame.parquet")
    m, _ = run(frame, cfg, seed, cfg["evaluation"]["bootstrap_reps"])
    (ROOT / "outputs" / "metrics.json").write_text(json.dumps(m, indent=1, sort_keys=True, ensure_ascii=False))
    (ROOT / "reports" / "figures").mkdir(parents=True, exist_ok=True)
    calibration_figure(m, ROOT / "reports" / "figures" / "calibration.png")


if __name__ == "__main__":
    main(sys.argv[1:])
