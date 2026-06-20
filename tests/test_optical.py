"""test_optical.py — 親機光学リンクの減衰・拡散モデル (MATH_SPEC §9) の数値テスト。

光は水中で減衰し、遠い/濁るほど SNR が落ちて角度精度が悪化、ついには見失う。
基準点での一致・単調性・ドロップアウト・後方互換を検証する。
"""
import numpy as np
import pytest

from src.sensors import (optical_signal_ratio, optical_snr, optical_angular_sigma,
                         optical_dropout_prob, simulate_observation,
                         simulate_observation_realistic)

# テスト用の光学モデル (config.OPTICAL_MODEL と同形, 角度は rad)
MODEL = {
    "attenuation_c": 0.30,
    "range_ref": 10.0,
    "sigma_ref": np.deg2rad(0.3),
    "sigma_floor": np.deg2rad(0.08),
    "snr_ref": 40.0,
    "snr_exponent": 1.0,
    "snr_min": 6.0,
    "dropout_max": 0.5,
    "dropout_jump": np.deg2rad(30.0),
}
SIGMA = (0.03, np.deg2rad(0.3), np.deg2rad(0.3))
P_PARENT = np.zeros(3)


# --- 9.(C)1  基準点 --------------------------------------------------------
def test_reference_point():
    dref = MODEL["range_ref"]
    assert optical_signal_ratio(dref, MODEL["attenuation_c"], dref) == pytest.approx(1.0, rel=1e-12)
    assert optical_snr(dref, MODEL) == pytest.approx(MODEL["snr_ref"], rel=1e-9)
    assert optical_angular_sigma(dref, MODEL) == pytest.approx(MODEL["sigma_ref"], rel=1e-9)


# --- 9.(C)2  単調性 --------------------------------------------------------
def test_snr_decreases_sigma_increases_with_range():
    snr_near = optical_snr(5.0, MODEL)
    snr_far = optical_snr(20.0, MODEL)
    assert snr_near > snr_far
    s_near = optical_angular_sigma(5.0, MODEL)
    s_far = optical_angular_sigma(20.0, MODEL)
    assert s_far > s_near
    assert s_near >= MODEL["sigma_floor"]            # 床を下回らない


def test_turbidity_worsens():
    clear = {**MODEL, "attenuation_c": 0.05}
    turbid = {**MODEL, "attenuation_c": 1.0}
    d = 15.0
    assert optical_snr(d, clear) > optical_snr(d, turbid)
    assert optical_angular_sigma(d, turbid) > optical_angular_sigma(d, clear)


def test_sigma_approaches_floor_when_close():
    s_close = optical_angular_sigma(1.0, MODEL)       # かなり近い -> 床近傍
    assert s_close < MODEL["sigma_ref"]
    assert s_close >= MODEL["sigma_floor"]


# --- 9.(C)3  ドロップアウト ------------------------------------------------
def test_dropout_in_range_and_monotonic():
    for d in [3, 5, 10, 15, 20, 30]:
        p = optical_dropout_prob(d, MODEL)
        assert 0.0 <= p <= MODEL["dropout_max"]
    assert optical_dropout_prob(30.0, MODEL) > optical_dropout_prob(5.0, MODEL)


def test_dropout_half_at_snr_min():
    """SNR=snr_min となる距離で p_drop = dropout_max/2。"""
    # SNR(d)=snr_ref*R^p=snr_min を解く: R = (snr_min/snr_ref)^(1/p)
    p = MODEL["snr_exponent"]
    R_target = (MODEL["snr_min"] / MODEL["snr_ref"]) ** (1.0 / p)
    # R(d)= (dref/d)^2 exp(-c(d-dref)) を数値で d を探す (単調減少)
    from scipy.optimize import brentq
    c, dref = MODEL["attenuation_c"], MODEL["range_ref"]
    f = lambda d: optical_signal_ratio(d, c, dref) - R_target
    d_star = brentq(f, dref, 100.0)
    assert optical_dropout_prob(d_star, MODEL) == pytest.approx(MODEL["dropout_max"] / 2, rel=1e-6)


# --- 9.(C)4  後方互換 ------------------------------------------------------
def test_optical_none_matches_baseline():
    """optical_model=None なら §8 既定 (= simulate_observation) と完全一致。"""
    truth = np.array([3.0, 2.0, -15.0])
    for seed in range(15):
        z0 = simulate_observation(truth, SIGMA, seed=seed, p_parent=P_PARENT)
        z1 = simulate_observation_realistic(truth, SIGMA, seed=seed, p_parent=P_PARENT)
        assert np.allclose(z0, z1, atol=1e-12)


# --- 9.(C)5  角度劣化 (統計) -----------------------------------------------
def test_deeper_increases_angular_spread():
    """濁った水で深い (遠い) ほど方位・仰角ノイズの実測ばらつきが増える。"""
    turbid = {**MODEL, "attenuation_c": 0.6, "dropout_max": 0.0}  # 見失いは切ってσだけ見る
    shallow = np.array([3.0, 2.0, -5.0])
    deep = np.array([3.0, 2.0, -20.0])
    N = 1500
    az_s = np.std([simulate_observation_realistic(shallow, SIGMA, seed=s, p_parent=P_PARENT,
                                                  optical_model=turbid)[1] for s in range(N)])
    az_d = np.std([simulate_observation_realistic(deep, SIGMA, seed=s, p_parent=P_PARENT,
                                                  optical_model=turbid)[1] for s in range(N)])
    assert az_d > az_s * 1.5
