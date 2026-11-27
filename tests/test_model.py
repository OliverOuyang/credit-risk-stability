"""Model helpers against reference implementations; report placeholders."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

from crs.model import Woe, auc, ks
from crs.report import render


def test_auc_ks_match_reference():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.1).astype(int)
    s = rng.normal(size=2000) + y
    assert auc(y, s) == pytest.approx(roc_auc_score(y, s))
    assert ks(y, s) == pytest.approx(ks_2samp(s[y == 1], s[y == 0]).statistic)


def test_woe_bins_missing_rare_and_unseen():
    X = pd.DataFrame({"classic_fico": [600.0, 650, 700, 750, np.nan] * 40,
                      "channel": ["R", "R", "C", "C", "T"] * 39 + ["Z"] * 5})
    y = pd.Series([1, 1, 0, 0, 1] * 40)
    w = Woe({"max_bins": 4, "min_bin_share": 0.05, "smoothing": 0.5}).fit(X, y)
    assert "_missing" in w.table["classic_fico"] and "_other" in w.table["channel"]  # Z is 2.5% -> _other
    t = w.transform(pd.DataFrame({"classic_fico": [600.0], "channel": ["never-seen"]}))
    assert t.loc[0, "channel"] == w.table["channel"]["_other"]
    assert w.iv["classic_fico"] > 0 and w.iv["channel"] > 0


def test_render_placeholders():
    m = {"a": {"auc": 0.77123, "cal": [{"n": 1234}]}}
    assert render("{{metrics.a.auc:.3f}} {{metrics.a.cal.0.n:,}} {{config.split.gap_months}}", m) == "0.771 1,234 24"
    with pytest.raises(KeyError):
        render("{{metrics.a.missing}}", m)
