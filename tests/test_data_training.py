"""check_data and check_training: green on a clean copy, red on deliberately broken data, rules or metrics."""
import json
import shutil

import pandas as pd
import pytest
import yaml

from checks import check_data, check_training
from checks.common import ROOT
import crs.target

TABLE = """| vintage | loans_raw | loans_uncensored | loans_main | bads_main | bads_incl_assisted | assisted |
|--:|--:|--:|--:|--:|--:|--:|
| 2010 | 3 | 3 | 2 | 1 | 2 | 1 |
"""


def lid(i):
    return "F10Q1" + f"{i:07d}"  # built here so the file holds no loan id


@pytest.fixture
def repo(tmp_path):
    shutil.copytree(ROOT / "config", tmp_path / "config")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "bad_counts_raw.md").write_text(TABLE)
    return tmp_path


def frame():
    """Three 2010 loans matching TABLE: one good, one bad, one bad under forbearance."""
    feats = [f["name"] for f in yaml.safe_load((ROOT / "config" / "features.yaml").read_text())["origination"]
             if f.get("use")]
    d = pd.DataFrame({c: ["N"] * 3 for c in feats})
    d = d.assign(loan_id=[lid(1), lid(2), lid(3)], vintage_year=2010, bad=[0, 1, 1], assisted=[0, 0, 1],
                 classic_fico=[700.0, 650.0, 720.0], mi_pct=0.0, number_of_units=1.0, original_cltv=80.0,
                 original_dti=30.0, original_ltv=80.0, number_of_borrowers=1.0, original_upb=200000.0,
                 original_interest_rate=5.0, original_loan_term=360.0)
    return d


def edit_yaml(root, name, fn):
    p = root / "config" / f"{name}.yaml"
    d = yaml.safe_load(p.read_text())
    fn(d)
    p.write_text(yaml.safe_dump(d, allow_unicode=True))


def run(repo, f=None):
    return check_data.run(repo, {"f": frame() if f is None else f}, raw_dir=repo / "no_raw")


def test_real_repo_is_green():
    assert check_data.run(ROOT) == []
    assert check_training.run(ROOT) == []


def test_clean_copy_is_green(repo):
    assert run(repo) == []


# label rules: prepaid, forbearance, censoring and window boundaries
@pytest.mark.parametrize("broken, case", [
    (lambda d: d["bad_if_any_in_window"]["zero_balance_codes"].append("01"), "prepaid early"),
    (lambda d: d["assistance"]["flag_if_any_in_window"]["borrower_assistance_plan"].append("R"), "repayment plan"),
    (lambda d: d["assistance"]["flag_if_any_in_window"].update(borrower_assistance_plan=[]), "forbearance in window"),
    (lambda d: d["exclude"].update(zero_balance_codes=[]), "defect"),
])
def test_label_rules_catch_broken_definition(repo, broken, case):
    edit_yaml(repo, "target", broken)
    assert any(case in v for v in run(repo))


def test_label_rules_catch_off_by_one_window(repo, monkeypatch):
    real = crs.target.label
    monkeypatch.setattr(check_data, "label", lambda o, p, c: real(o, p, {**c, "performance_months": 25}))
    v = run(repo)
    assert any("one month after the window" in x for x in v)
    assert any("exactly at cutoff" in x for x in v)


# frame vs committed table, ids, ranges, missing
def test_rejects_row_dropped_vs_table(repo):
    assert any("loans_main 1" in v for v in run(repo, frame().iloc[1:]))


def test_rejects_bad_count_vs_table(repo):
    f = frame()
    f.loc[0, "bad"] = 1
    assert any("bads_main 2" in v for v in run(repo, f))


def test_rejects_duplicate_and_malformed_ids(repo):
    f = frame()
    f.loc[1, "loan_id"] = lid(1)
    f.loc[2, "loan_id"] = "X" + lid(3)[1:]
    v = run(repo, f)
    assert any("duplicate" in x for x in v) and any("not matching" in x for x in v)


def test_rejects_out_of_range_and_upb(repo):
    f = frame()
    f.loc[0, "classic_fico"] = 9999.0  # "not available" code left in
    f.loc[1, "original_upb"] = 200500.0
    v = run(repo, f)
    assert any("classic_fico" in x and "outside" in x for x in v)
    assert any("multiple of 1000" in x for x in v)


def test_rejects_high_missing_and_absent_feature(repo):
    f = frame().drop(columns="channel")
    f["original_dti"] = [None, None, 30.0]
    v = run(repo, f)
    assert any("original_dti missing" in x for x in v)
    assert any("channel missing from the frame" in x for x in v)


def test_rejects_raw_row_count_vs_table(repo):
    raw = repo / "raw" / "sample_2010"
    raw.mkdir(parents=True)
    row = lambda i: "|".join(["x"] * 19 + [lid(i)] + ["x"] * 11)
    (raw / "sample_orig_2010.txt").write_text("\n".join(row(i) for i in (1, 2)) + "\n")  # table says 3
    v = check_data.run(repo, {}, raw_dir=repo / "raw")
    assert any("origination rows" in x for x in v)


# training
def metrics(repo, **kw):
    t = {"lightgbm_best_iteration": 37, "lightgbm_max_rounds": 2000,
         "auc_gap_train_minus_valid": {"scorecard": 0.057, "lightgbm": 0.066},
         "auc_gap_excess_lightgbm_over_scorecard": 0.009}
    t.update(kw)
    (repo / "outputs").mkdir(exist_ok=True)
    (repo / "outputs" / "metrics.json").write_text(json.dumps({"training": t}))
    return check_training.run(repo)


def test_training_green(repo):
    assert metrics(repo) == []


@pytest.mark.parametrize("kw, word", [
    ({"lightgbm_best_iteration": 2000}, "never fired"),
    ({"lightgbm_best_iteration": 3}, "peaked at once"),
    ({"auc_gap_train_minus_valid": {"scorecard": 0.05, "lightgbm": 0.15}}, "overfitting or"),
    ({"auc_gap_train_minus_valid": {"scorecard": -0.05, "lightgbm": 0.0}}, "leakage"),
    ({"auc_gap_excess_lightgbm_over_scorecard": 0.05}, "exceeds the scorecard"),
])
def test_training_rejects(repo, kw, word):
    assert any(word in v for v in metrics(repo, **kw))


def test_training_rejects_missing_block(repo):
    (repo / "outputs").mkdir()
    (repo / "outputs" / "metrics.json").write_text("{}")
    assert any("no training block" in v for v in check_training.run(repo))
