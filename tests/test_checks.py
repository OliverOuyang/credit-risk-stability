"""Each check: green on the real repo, red on a deliberately broken copy."""
import json
import shutil
import sys

import pandas as pd
import pytest
import yaml

from checks import check_leakage, check_numbers, check_redline, check_repro, check_split
from checks.common import ROOT

# Built by concatenation so this file itself never contains a loan id or a secret.
LOAN = "F" + "13Q1" + "0000001"
LOAN2 = "F" + "18Q3" + "0000002"


@pytest.fixture
def repo(tmp_path):
    shutil.copytree(ROOT / "config", tmp_path / "config")
    (tmp_path / "docs").mkdir()
    shutil.copy(ROOT / "docs" / "bad_counts_raw.md", tmp_path / "docs")
    (tmp_path / "uv.lock").write_text("")
    return tmp_path


def edit_yaml(root, name, fn):
    p = root / "config" / f"{name}.yaml"
    d = yaml.safe_load(p.read_text()) or {}
    fn(d)
    p.write_text(yaml.safe_dump(d, allow_unicode=True))


def write(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return rel


@pytest.mark.parametrize("mod", [check_leakage, check_split, check_repro, check_numbers])
def test_real_repo_is_green(mod):
    assert mod.run(ROOT) == []


def test_real_repo_redline_green():
    assert check_redline.run(ROOT, check_redline.repo_files(ROOT)) == []


# 1 leakage
def test_leakage_rejects_non_origination_feature(repo):
    edit_yaml(repo, "features", lambda d: d["origination"][0].update(available_at="performance"))
    assert any("classic_fico" in v for v in check_leakage.run(repo))


def test_leakage_rejects_performance_column_in_frame(repo):
    frame = pd.DataFrame({"loan_id": [LOAN] * 4, "segment": "train", "bad": [0, 1, 0, 1],
                          "classic_fico": [700, 600, 720, 610], "current_upb": [1, 2, 3, 4]})
    assert any("current_upb" in v for v in check_leakage.run(repo, frame))


def test_leakage_flags_high_auc_until_allowlisted(repo):
    frame = pd.DataFrame({"loan_id": [LOAN] * 6, "segment": "train", "bad": [0, 0, 0, 1, 1, 1],
                          "original_ltv": [60, 65, 70, 95, 96, 97]})
    assert any("original_ltv" in v for v in check_leakage.run(repo, frame))
    edit_yaml(repo, "leakage_allowlist", lambda d: d.update(allow=[
        {"feature": "original_ltv", "confirmed_by": "oliver", "reason": "test", "date": "2026-11-27"}]))
    assert check_leakage.run(repo, frame) == []


def test_leakage_rejects_incomplete_allowlist_entry(repo):
    edit_yaml(repo, "leakage_allowlist", lambda d: d.update(allow=[{"feature": "original_ltv"}]))
    assert any("allowlist" in v for v in check_leakage.run(repo))


# 2 split
def test_split_rejects_missing_gap(repo):
    edit_yaml(repo, "split", lambda d: d["segments"]["valid"].update(start="2012Q1"))
    assert any("gap" in v for v in check_split.run(repo))


def test_split_rejects_test_too_close_to_cutoff(repo):
    edit_yaml(repo, "split", lambda d: d["segments"]["test"].update(end="2025Q1"))
    assert any("cutoff" in v for v in check_split.run(repo))


def test_split_rejects_per_loan_censoring(repo):
    edit_yaml(repo, "target", lambda d: d["censoring"].update(rule="observed_months"))
    assert any("calendar" in v for v in check_split.run(repo))


def test_split_rejects_thin_eval_segment(repo):
    edit_yaml(repo, "split", lambda d: d.update(eval_segments=[
        {"name": "2020", "start": "2020Q1", "end": "2020Q4"},
        {"name": "2021-2024", "start": "2021Q1", "end": "2024Q2"}]))
    assert any("2020" in v and "300" in v for v in check_split.run(repo))


def test_split_rejects_bad_frame(repo):
    frame = pd.DataFrame({"loan_id": [LOAN, LOAN, LOAN2],
                          "segment": ["valid", "test", "train"],      # overlap; 2018 loan in train
                          "first_payment_date": ["201303", "201303", "202508"],  # 2025-08+23m > cutoff
                          "bad": [0, 0, 1]})
    v = check_split.run(repo, frame)
    assert any("more than one segment" in x for x in v)
    assert any(LOAN2 in x and "train" in x for x in v)
    assert any("censor" in x for x in v)


# 3 repro
def dev_cmd(out, expr):
    return f'"{sys.executable}" -c "import json,random; json.dump({{\'auc\': {expr}}}, open(\'{out}\', \'w\'))"'


def test_repro_rejects_nondeterministic_dev_run(repo):
    out = repo / "dev.json"
    edit_yaml(repo, "repro", lambda d: d.update(dev_command=dev_cmd(out, "random.random()"),
                                                  dev_output="dev.json"))
    assert any("auc" in v for v in check_repro.run(repo))


def test_repro_accepts_deterministic_and_checks_reference(repo):
    out = repo / "dev.json"
    edit_yaml(repo, "repro", lambda d: d.update(dev_command=dev_cmd(out, "0.75"), dev_output="dev.json"))
    assert check_repro.run(repo) == []
    write(repo, "outputs/dev_metrics_reference.json", json.dumps({"auc": 0.7}))
    edit_yaml(repo, "repro", lambda d: d.update(reference="outputs/dev_metrics_reference.json"))
    assert any("reference" in v for v in check_repro.run(repo))


def test_repro_rejects_missing_lock(repo):
    (repo / "uv.lock").unlink()
    assert any("uv.lock" in v for v in check_repro.run(repo))


# 4 numbers
def test_numbers_sources_and_skips(repo):
    write(repo, "outputs/metrics.json", json.dumps({"auc_test": 0.7812, "approval_rate": 0.634}))
    write(repo, "reports/ok.md", "## 3.1 结论\n\n1. 跨期 AUC 为 0.78，批准率 63.4%。\n"
                                 "2. 表现期 24 个月，2015 年发放，2005–2010 训练，PSI 超过 0.25 要重训。\n"
                                 "```\nx = 12345\n```\n占位 {{metrics.auc_test}}，代码 `n=999`。\n")
    assert check_numbers.run(repo) == []


def test_numbers_rejects_unsourced(repo):
    write(repo, "outputs/metrics.json", json.dumps({"auc_test": 0.7812}))
    write(repo, "reports/bad.md", "测试段 AUC 0.81。\n\n共 2015 笔贷款。\n")
    v = check_numbers.run(repo)
    assert any("bad.md:1" in x and "0.81" in x for x in v)
    assert any("bad.md:3" in x and "2015" in x for x in v)


def test_numbers_scans_site_html(repo):
    write(repo, "docs/site/index.html", '<p style="width:300px">批准提升 7.7%</p>')
    v = check_numbers.run(repo)
    assert len(v) == 1 and "7.7" in v[0]


# 5 redline
def test_redline_rejects_paths(repo):
    files = [write(repo, "data/raw/sample_2010/sample_orig_2010.txt", "x"),
             write(repo, "data/interim/frame.parquet", "x"),
             write(repo, "notebooks/loans.csv", "a,b\n1,2\n"),
             write(repo, "docs/historical_data_2010Q1.txt", "x")]
    v = check_redline.run(repo, files)
    assert all(any(f in x for x in v) for f in files)


def test_redline_allows_allowlisted_paths(repo):
    files = [write(repo, "docs/workflow_log.csv", "issue,runs\nMULT-1,2\n"),
             write(repo, "app/data/agg/psi.csv", "quarter,n,psi\n2010Q1,500,0.02\n"),
             write(repo, "data/manifest.sha256", "abc  ./sample_2010.zip\n")]
    assert check_redline.run(repo, files) == []


def test_redline_rejects_bad_agg_table(repo):
    f1 = write(repo, "app/data/agg/small.csv", "quarter,n\n2010Q1,3\n")
    f2 = write(repo, "app/data/agg/ids.csv", f"loan_id,n\n{LOAN},500\n")
    f3 = write(repo, "app/data/agg/nocount.csv", "quarter,psi\n2010Q1,0.1\n")
    v = check_redline.run(repo, [f1, f2, f3])
    assert all(any(f in x for x in v) for f in (f1, f2, f3))


def test_redline_rejects_loan_id_in_report(repo):
    f = write(repo, "reports/r.md", f"例如 {LOAN} 这笔贷款\n")
    assert any("r.md:1" in x for x in check_redline.run(repo, [f]))


def test_redline_rejects_raw_rows(repo):
    row = "|".join(["x"] * 31)
    f = write(repo, "reports/sample.md", "\n".join([row] * 21))
    assert any("sample.md" in x for x in check_redline.run(repo, [f]))


def test_redline_rejects_notebook_outputs(repo):
    nb = {"cells": [{"cell_type": "code", "source": ["1+1"], "outputs": [{"output_type": "stream", "text": ["2"]}]}]}
    f = write(repo, "notebooks/eda.ipynb", json.dumps(nb))
    assert any("eda.ipynb" in x for x in check_redline.run(repo, [f]))


def test_redline_rejects_secret(repo):
    f = write(repo, "src/x.py", "KEY = 'gh" + "p_" + "a" * 36 + "'\n")
    assert any("x.py:1" in x for x in check_redline.run(repo, [f]))


def test_redline_rejects_blacklisted_keyword(repo, tmp_path_factory, monkeypatch):
    bl = tmp_path_factory.mktemp("bl") / "blacklist.txt"
    bl.write_text("# private\nacmecorp\n")
    monkeypatch.setenv("CRS_REDLINE_BLACKLIST", str(bl))
    f = write(repo, "reports/r.md", "参考 AcmeCorp 的口径\n")
    assert any("r.md:1" in x for x in check_redline.run(repo, [f]))
