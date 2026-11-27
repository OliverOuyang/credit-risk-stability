"""Cross-vintage stability of the 2c models: quarterly PSI / CSI / AUC, fixed vs annual rolling retrain,
crisis and forbearance sensitivity, recalibration rules backtest. Needs data/interim/model_frame.parquet and
stability_frame.parquet (`make frame`, `make stability-frame`). Writes outputs/stability.json,
app/data/agg/*.csv and reports/figures/stability_*.png."""
import json

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from crs.frame import NUMERIC, loan_quarter_index, quarter_index
from crs.model import R, Woe, auc_ks, features, fit_lightgbm, fit_scorecard, interval
from crs.target import ROOT, load_cfg

MODELS = ("scorecard", "lightgbm")
EPS = 1e-6  # floor for empty bins in PSI


def psi(expected, actual):
    """Population stability index between two share vectors over the same bins."""
    e, a = np.clip(expected, EPS, None), np.clip(actual, EPS, None)
    return float(((a - e) * np.log(a / e)).sum())


def shares(bins, labels):
    return pd.Series(bins).value_counts(normalize=True).reindex(labels, fill_value=0).to_numpy()


def score_bins(train_score, n=10):
    """Train-decile edges; returns a function score -> bin index."""
    edges = np.unique(np.quantile(train_score, np.linspace(0, 1, n + 1)[1:-1]))
    return lambda s: np.searchsorted(edges, s, side="right"), list(range(len(edges) + 1))


def quarter_label(qi):
    return f"{qi // 4}Q{qi % 4 + 1}"


def period_of(qi, periods):
    out = pd.Series("normal", index=qi.index)
    for k, p in periods.items():
        out[qi.between(quarter_index(p["start"]), quarter_index(p["end"]))] = k
    return out


def calib(y, p, level):
    """Mean prediction, bad rate, Brier, and z of observed vs expected bads (fires outside the ci_level band)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    z = (y.sum() - p.sum()) / np.sqrt((p * (1 - p)).sum())
    return {"mean_pred": round(float(p.mean()), R), "bad_rate": round(float(y.mean()), R),
            "brier": round(float(np.mean((p - y) ** 2)), R), "oe": round(float(y.sum() / p.sum()), R),
            "z": round(float(z), R), "miscalibrated": bool(abs(z) > norm.ppf((1 + level) / 2))}


def boot_auc(y, scores, pairs, reps, rng, level):
    """AUC intervals per score and paired differences (a - b) on shared resamples."""
    y = np.asarray(y, int)
    est = {k: auc_ks(y, s)[0] for k, s in scores.items()}
    boot = {k: [] for k in scores}
    for _ in range(reps):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            for k, s in scores.items():
                boot[k].append(auc_ks(y[i], s[i])[0])
    out = {k: interval(est[k], boot[k], level) for k in scores}
    for a, b in pairs:
        out[f"{a}-{b}"] = interval(est[a] - est[b], np.subtract(boot[a], boot[b]), level)
    return out


def platt(p_fit, y_fit):
    """Logistic recalibration on logit(p), fitted on the latest labelled vintage."""
    lg = lambda p: np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS))).reshape(-1, 1)
    lr = LogisticRegression(C=1e6).fit(lg(p_fit), y_fit)  # C large = unpenalised
    return lambda p: lr.predict_proba(lg(p))[:, 1]


def fit_both(train, valid, mcfg, seed):
    sc, sc_info = fit_scorecard(train, mcfg)
    gb, gb_info = fit_lightgbm(train, valid, mcfg, seed)
    return {"scorecard": sc, "lightgbm": gb}, sc_info


def band(value, cuts):
    return int(np.searchsorted(cuts, value, side="right"))  # 0 stable, 1 watch, 2 act


def first_sustained(bands, labels, level, k):
    """First quarter where band >= level for k quarters in a row (reported at the k-th quarter)."""
    run = 0
    for b, q in zip(bands, labels):
        run = run + 1 if b >= level else 0
        if run >= k:
            return q
    return None


def quarterly(wide, preds, sc_bins, kept, woe, base, cfg):
    """One row per origination quarter plus a long CSI table; PSI/CSI against the train segment."""
    level = load_cfg("model")["evaluation"]["ci_level"]
    rows, csi_rows, sh = [], [], {m: {} for m in MODELS}
    for qi, d in wide.groupby("qi"):
        main = d[d.assisted == 0]
        y = main["bad"].to_numpy()
        r = {"quarter": quarter_label(qi), "period": main["period"].iat[0], "segment": main["segment"].iat[0],
             "n": len(main), "bads": int(y.sum()), "bad_rate": round(float(y.mean()), R),
             "n_incl_assisted": len(d), "bad_rate_incl_assisted": round(float(d["bad"].mean()), R)}
        for m in MODELS:
            p = preds[m][main.index].to_numpy()
            sh[m][qi] = shares(sc_bins[m][0](p), sc_bins[m][1])
            r[f"psi_{m}"] = round(psi(base[m], sh[m][qi]), R)
            # year-over-year: same quarter one year earlier as the baseline (train-decile bins)
            r[f"psi_yoy_{m}"] = round(psi(sh[m][qi - 4], sh[m][qi]), R) if qi - 4 in sh[m] else None
            r[f"mean_pred_{m}"] = round(float(p.mean()), R)
            r[f"auc_{m}"] = round(auc_ks(y, p)[0], R) if y.sum() >= cfg["min_bads_for_auc"] else None
            r[f"oe_{m}"] = calib(y, p, level)["oe"]
        for c in kept:
            b = woe._bin(c, main[c])
            csi_rows.append({"quarter": r["quarter"], "feature": c, "n": len(main),
                             "csi": round(psi(base["csi"][c], shares(b, list(base["csi_labels"][c]))), R)})
        rows.append(r)
    return pd.DataFrame(rows), pd.DataFrame(csi_rows)


def annual(wide, model_frame, fixed, mcfg, cfg, seed, valid_auc):
    """Per backtest year: fixed, recalibrated fixed, rolling retrain, no-crisis fixed, fixed incl. assisted."""
    level = mcfg["evaluation"]["ci_level"]
    rng = np.random.default_rng(seed)
    main = wide[wide.assisted == 0]
    vy = main["vintage_year"]
    mf = model_frame[model_frame.assisted == 0]
    drop = cfg["crisis_sensitivity"]["drop_years"]
    nc, _ = fit_both(mf[(mf.segment == "train") & ~mf.vintage_year.isin(drop)], mf[mf.segment == "valid"], mcfg, seed)
    lo, hi = cfg["rolling"]["train_years_back"]
    out = {}
    for Y in cfg["backtest_years"]:
        print(f"[stability] backtest {Y}", flush=True)
        d, da = main[vy == Y], wide[(wide.vintage_year == Y) & (wide.segment != "gap")]
        roll, _ = fit_both(main[vy.between(Y - lo, Y - hi)], main[vy == Y - cfg["rolling"]["valid_year_back"]],
                           mcfg, seed)
        fit_d = main[vy == Y - cfg["recalibration"]["fit_year_back"]]
        res = {"sample": {"loans": len(d), "bads": int(d.bad.sum()), "bad_rate": round(float(d.bad.mean()), R),
                          "loans_incl_assisted": len(da), "bad_rate_incl_assisted": round(float(da.bad.mean()), R)}}
        for m in MODELS:
            v = {"fixed": fixed[m](d), "rolling": roll[m](d), "no_crisis": nc[m](d)}
            v["recal"] = platt(fixed[m](fit_d), fit_d["bad"])(v["fixed"])
            a = boot_auc(d["bad"], v, [("rolling", "fixed"), ("no_crisis", "fixed")], cfg["bootstrap_reps"], rng, level)
            res[m] = {k: {"auc": a[k], **calib(d["bad"], p, level)} for k, p in v.items()}
            res[m]["delta_rolling_minus_fixed"] = a["rolling-fixed"]
            res[m]["delta_no_crisis_minus_fixed"] = a["no_crisis-fixed"]
            pa = fixed[m](da)
            res[m]["incl_assisted"] = {"auc": round(auc_ks(da["bad"].to_numpy(), pa)[0], R),
                                       **calib(da["bad"], pa, level)}
            f = res[m]["fixed"]
            res[m]["rule"] = {"recalibrate": f["miscalibrated"], "retrain": bool(f["auc"]["hi"] < valid_auc[m]["lo"])}
        out[str(Y)] = res
    return out


def pooled_periods(wide, preds, sc_bins, base, level):
    out = {}
    for k, d in wide[wide.assisted == 0].groupby("period"):
        out[k] = {"loans": len(d), "bads": int(d.bad.sum()), "bad_rate": round(float(d.bad.mean()), R)}
        for m in MODELS:
            p = preds[m][d.index].to_numpy()
            out[k][m] = {"auc": round(auc_ks(d["bad"].to_numpy(), p)[0], R),
                         "psi": round(psi(base[m], shares(sc_bins[m][0](p), sc_bins[m][1])), R), **calib(d["bad"], p, level)}
    return out


def figures(q, csi, ann, psi_cuts, periods, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    color = {"scorecard": "#2563eb", "lightgbm": "#ea580c"}
    x = np.arange(len(q))
    ticks = [i for i, s in enumerate(q.quarter) if s.endswith("Q1") and int(s[:4]) % 2 == 1]

    def shade(ax):
        for k, p in periods.items():
            s = q.index[(q.quarter >= p["start"]) & (q.quarter <= p["end"])]
            ax.axvspan(s.min() - 0.5, s.max() + 0.5, color="#e5e7eb", zorder=0)
            ax.text((s.min() + s.max()) / 2, ax.get_ylim()[1], k, ha="center", va="top", fontsize=8, color="#6b7280")
        ax.set_xticks(ticks, [q.quarter[i][:4] for i in ticks], fontsize=8)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True)
    for m in MODELS:
        axes[0].plot(x, q[f"psi_{m}"], "o-", ms=3, color=color[m], label=f"{m} vs train")
        axes[0].plot(x, q[f"psi_yoy_{m}"].astype(float), "x--", ms=3, color=color[m], alpha=0.6,
                     label=f"{m} vs same quarter last year")
        axes[1].plot(x, q[f"auc_{m}"].astype(float), "o-", ms=3, color=color[m], label=m)
    for c in psi_cuts:
        axes[0].axhline(c, ls=":", color="grey")
    axes[0].set_ylabel("score PSI")
    axes[0].set_yscale("log")
    axes[1].set_ylabel("AUC (quarter)")
    axes[0].legend()
    for ax in axes:
        shade(ax)
    fig.tight_layout()
    fig.savefig(path / "stability_quarterly.png", dpi=120)

    h = csi.pivot(index="feature", columns="quarter", values="csi")
    h = h.loc[h.max(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(10, 0.4 * len(h) + 1.2))
    im = ax.imshow(h.to_numpy(), aspect="auto", cmap="Oranges", vmin=0, vmax=max(psi_cuts))
    ax.set_yticks(range(len(h)), h.index, fontsize=8)
    cols = list(h.columns)
    tk = [i for i, s in enumerate(cols) if s.endswith("Q1") and int(s[:4]) % 2 == 1]
    ax.set_xticks(tk, [cols[i][:4] for i in tk], fontsize=8)
    fig.colorbar(im, ax=ax, label="CSI vs train (capped at the act threshold)")
    fig.tight_layout()
    fig.savefig(path / "stability_csi.png", dpi=120)

    years = list(ann)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True)
    for j, m in enumerate(MODELS):
        for k, st in (("fixed", "o-"), ("rolling", "s--"), ("recal", "^:")):
            if k != "recal":
                axes[0, j].plot(years, [ann[y][m][k]["auc"]["est"] for y in years], st, color=color[m], label=k, ms=4)
            axes[1, j].plot(years, [ann[y][m][k]["mean_pred"] for y in years], st, color=color[m], label=k, ms=4)
        axes[1, j].plot(years, [ann[y]["sample"]["bad_rate"] for y in years], "k-", lw=2, label="observed")
        axes[0, j].set_title(m)
        axes[0, j].legend(fontsize=8)
        axes[1, j].legend(fontsize=8)
    axes[0, 0].set_ylabel("AUC (vintage year)")
    axes[1, 0].set_ylabel("mean predicted PD vs bad rate")
    fig.tight_layout()
    fig.savefig(path / "stability_annual.png", dpi=120)


def main():
    cfg, mcfg = load_cfg("stability"), load_cfg("model")
    seed, level = load_cfg("repro")["seed"], mcfg["evaluation"]["ci_level"]
    psi_cuts = next(c["values"] for c in load_cfg("constants_allowlist")["constants"] if c.get("id") == "psi")
    mf = pd.read_parquet(ROOT / "data" / "interim" / "model_frame.parquet")
    wide = pd.read_parquet(ROOT / "data" / "interim" / "stability_frame.parquet")
    wide["qi"] = loan_quarter_index(wide["loan_id"])
    wide["period"] = period_of(wide["qi"], cfg["periods"])
    m_main = mf[mf.assisted == 0]
    train = m_main[m_main.segment == "train"]
    fixed, sc_info = fit_both(train, m_main[m_main.segment == "valid"], mcfg, seed)

    metrics = json.loads((ROOT / "outputs" / "metrics.json").read_text())
    test = m_main[m_main.segment == "test"]
    for m in MODELS:  # refit must reproduce 2c exactly
        a = auc_ks(test["bad"].to_numpy(), fixed[m](test))[0]
        assert abs(a - metrics[m]["segments"]["test"]["auc"]["est"]) < 1e-5, (m, a)

    preds = {m: pd.Series(fixed[m](wide), index=wide.index) for m in MODELS}
    train_p = {m: fixed[m](train) for m in MODELS}
    sc_bins = {m: score_bins(train_p[m]) for m in MODELS}
    kept = [c for c, v in sc_info["features"].items() if v["kept"]]
    woe = Woe(mcfg["woe"]).fit(train[features()], train["bad"])
    base = {m: shares(sc_bins[m][0](train_p[m]), sc_bins[m][1]) for m in MODELS}
    base["csi_labels"] = {c: sorted(set(woe._bin(c, train[c])) | {"_other", "_missing"}) for c in kept}
    base["csi"] = {c: shares(woe._bin(c, train[c]), base["csi_labels"][c]) for c in kept}

    q, csi = quarterly(wide, preds, sc_bins, kept, woe, base, cfg)
    valid_auc = {m: metrics[m]["segments"]["valid"]["auc"] for m in MODELS}
    ann = annual(wide, mf, fixed, mcfg, cfg, seed, valid_auc)

    test_q = q[q.segment == "test"]
    rules = {}
    for m in MODELS:
        bands = [band(v, psi_cuts) for v in q[f"psi_{m}"]]
        yoy = [band(v, psi_cuts) if v == v and v is not None else 0 for v in q[f"psi_yoy_{m}"]]
        ty = yoy[-len(test_q):]
        k = cfg["rules"]["consecutive_quarters"]
        rules[m] = {"yoy_test_quarters_watch": int(sum(b >= 1 for b in ty)),
                    "yoy_test_quarters_act": int(sum(b >= 2 for b in ty)),
                    "yoy_fired_act": [q.quarter.iat[i] for i in range(len(q)) if i >= k - 1
                                      and all(b >= 2 for b in yoy[i - k + 1:i + 1])],
                    "yoy_fired_watch": [q.quarter.iat[i] for i in range(len(q)) if i >= k - 1
                                        and all(b >= 1 for b in yoy[i - k + 1:i + 1])],
                    "years_rolling_better": [y for y in ann if ann[y][m]["delta_rolling_minus_fixed"]["lo"] > 0],
                    "years_no_crisis_better": [y for y in ann if ann[y][m]["delta_no_crisis_minus_fixed"]["lo"] > 0],
                    "years_recal_still_off": [y for y in ann if ann[y][m]["recal"]["miscalibrated"]],
                    "years_recal_brier_better": [y for y in ann if ann[y][m]["recal"]["brier"] < ann[y][m]["fixed"]["brier"]],
"first_watch": first_sustained(bands, q.quarter, 1, cfg["rules"]["consecutive_quarters"]),
                    "first_act": first_sustained(bands, q.quarter, 2, cfg["rules"]["consecutive_quarters"]),
                    "test_quarters_watch": int(sum(b >= 1 for b in bands[-len(test_q):])),
                    "test_quarters_act": int(sum(b >= 2 for b in bands[-len(test_q):])),
                    "psi_max": round(float(q[f"psi_{m}"].max()), R),
                    "psi_max_quarter": q.quarter[q[f"psi_{m}"].idxmax()],
                    "years_recalibrate": [y for y in ann if ann[y][m]["rule"]["recalibrate"]],
                    "years_retrain": [y for y in ann if ann[y][m]["rule"]["retrain"]]}
    csi_peak = csi.loc[csi.groupby("feature")["csi"].idxmax()].sort_values("csi", ascending=False)
    out = {"quarters": {"n": len(q), "first": q.quarter.iat[0], "last": q.quarter.iat[-1],
                        "test": len(test_q), "min_loans": int(q.n.min())},
           "psi_thresholds": psi_cuts, "rules": rules, "annual": ann,
           "periods": pooled_periods(wide, preds, sc_bins, base, level),
           "csi_peak": {r.feature: {"csi": r.csi, "quarter": r.quarter} for r in csi_peak.itertuples()},
           "csi_mean_test": {c: round(float(v), R) for c, v in
                             csi[csi.quarter.isin(test_q.quarter)].groupby("feature")["csi"].mean().items()}}
    (ROOT / "outputs" / "stability.json").write_text(json.dumps(out, indent=1, sort_keys=True, ensure_ascii=False))

    agg = ROOT / "app" / "data" / "agg"
    agg.mkdir(parents=True, exist_ok=True)
    q.to_csv(agg / "quarterly.csv", index=False)
    csi.to_csv(agg / "csi_quarterly.csv", index=False)
    pd.DataFrame([{"vintage_year": int(y), "model": m, "variant": k, "n": ann[y]["sample"]["loans"],
                   "bads": ann[y]["sample"]["bads"], "auc": v["auc"]["est"], "auc_lo": v["auc"]["lo"],
                   "auc_hi": v["auc"]["hi"], **{c: v[c] for c in ("mean_pred", "bad_rate", "brier", "oe", "z")}}
                  for y in ann for m in MODELS for k in ("fixed", "recal", "rolling", "no_crisis")
                  for v in [ann[y][m][k]]]).to_csv(agg / "annual.csv", index=False)
    figures(q, csi, ann, psi_cuts, cfg["periods"], ROOT / "reports" / "figures")


if __name__ == "__main__":
    main()
