import pandas as pd

from crs.target import label, load_cfg


def perf(loan, rows):
    return pd.DataFrame([dict(loan_id=loan, period="", dlq=d, loan_age=str(a), zbc=z,
                              deferral=None, disaster=None, assistance=s) for a, d, z, s in rows])


def test_label_rules():
    cfg = load_cfg("target")
    orig = pd.DataFrame({"loan_id": ["F10Q1000001", "F10Q1000002", "F10Q1000003", "F10Q1000004",
                                     "F10Q1000005", "F24Q2000006"],
                         "first_payment_date": ["201003"] * 5 + ["202408"]})
    p = pd.concat([
        perf("F10Q1000001", [(5, "0", None, None), (12, "3", None, None)]),     # 90+ dpd -> bad
        perf("F10Q1000002", [(6, "0", "01", None)]),                             # prepaid -> good, kept
        perf("F10Q1000003", [(30, "5", None, None)]),                            # bad after month 24 -> good
        perf("F10Q1000004", [(8, "1", None, "F")]),                              # forbearance -> assisted
        perf("F10Q1000005", [(10, "RA", None, None)]),                           # REO acquisition -> bad
        perf("F24Q2000006", [(1, "0", None, None)]),                             # fpd+23 > 2026-06 -> censored
    ])
    out = label(orig, p, cfg).set_index("loan_id")
    assert "F24Q2000006" not in out.index
    assert out.loc["F10Q1000001", "bad"] == 1
    assert out.loc["F10Q1000002", "bad"] == 0
    assert out.loc["F10Q1000003", "bad"] == 0
    assert out.loc["F10Q1000004", "assisted"] == 1
    assert out.loc["F10Q1000005", "bad"] == 1
    assert (out["vintage_year"] == 2010).all()
