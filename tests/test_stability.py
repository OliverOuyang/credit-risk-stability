"""Stability helpers: PSI, sustained-band rule, calibration z, recalibration."""
import numpy as np
import pytest

from crs.stability import band, calib, first_sustained, platt, psi, score_bins, shares


def test_psi_zero_for_same_and_known_value():
    e = np.array([0.5, 0.5])
    assert psi(e, e) == 0
    a = np.array([0.6, 0.4])
    assert psi(e, a) == pytest.approx(0.1 * np.log(1.2) - 0.1 * np.log(0.8))


def test_score_bins_train_deciles_have_equal_shares():
    s = np.random.default_rng(0).random(10000)
    f, labels = score_bins(s)
    assert np.allclose(shares(f(s), labels), 0.1, atol=1e-3)


def test_rule_needs_consecutive_quarters():
    cuts = [0.1, 0.25]
    bands = [band(v, cuts) for v in [0.05, 0.12, 0.03, 0.3, 0.26, 0.11]]
    assert bands == [0, 1, 0, 2, 2, 1]
    q = list("abcdef")
    assert first_sustained(bands, q, 1, 2) == "e"   # b alone is not sustained
    assert first_sustained(bands, q, 2, 2) == "e"
    assert first_sustained(bands, q, 2, 3) is None


def test_calibration_flags_overprediction_and_platt_fixes_it():
    rng = np.random.default_rng(1)
    p_true = rng.uniform(0.001, 0.02, 50000)
    y = (rng.random(50000) < p_true).astype(int)
    assert not calib(y, p_true, 0.95)["miscalibrated"]
    high = np.clip(p_true * 2.5, 0, 1)
    c = calib(y, high, 0.95)
    assert c["miscalibrated"] and c["z"] < 0 and c["oe"] < 0.5
    y2 = (rng.random(50000) < p_true).astype(int)
    assert not calib(y2, platt(high, y)(high), 0.95)["miscalibrated"]
