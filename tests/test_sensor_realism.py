"""test_sensor_realism.py — 現実的センサ誤差モデル (MATH_SPEC §8) の数値テスト。

§8 は §1〜§7 の理想モデルへ重ねる追加項。既定値はすべて『理想』で、
従来の simulate_observation と完全一致することをまず保証する (後方互換)。
"""
import numpy as np
import pytest

from src.sensors import (forward_observation, relative_vector,
                         simulate_observation, simulate_observation_realistic,
                         simulate_observation_sequence,
                         simulate_observation_sequence_realistic,
                         effective_sigma)

SIGMA = (0.03, np.deg2rad(0.3), np.deg2rad(0.3))
P = np.array([6.0, 8.0, -7.5])          # d_true = 12.5 m
P_PARENT = np.zeros(3)


# --- 8.(C)1  理想一致 (後方互換) -------------------------------------------
def test_defaults_match_ideal():
    """全既定なら simulate_observation_realistic は simulate_observation と一致。"""
    for seed in range(20):
        z_ideal = simulate_observation(P, SIGMA, seed=seed, p_parent=P_PARENT)
        z_real = simulate_observation_realistic(P, SIGMA, seed=seed, p_parent=P_PARENT)
        assert np.allclose(z_ideal, z_real, atol=1e-12)


def test_sequence_defaults_match_ideal():
    traj = np.array([[3.0, 4.0, -5.0], [3.5, 4.2, -5.1], [4.0, 4.4, -5.2]])
    z_ideal = simulate_observation_sequence(traj, SIGMA, seed=7, p_parent=P_PARENT)
    # velocity による距離の時刻ズレは latency=0 (既定) では効かない
    z_real = simulate_observation_sequence_realistic(traj, SIGMA, seed=7,
                                                     p_parent=P_PARENT)
    assert np.allclose(z_ideal, z_real, atol=1e-12)


# --- 8.(C)2  音速スケール (ノイズ0) ----------------------------------------
def test_sound_speed_scale_noise_free():
    zero = (0.0, 0.0, 0.0)
    d_true = np.linalg.norm(P)
    z = simulate_observation_realistic(P, zero, seed=0, p_parent=P_PARENT,
                                       sound_speed_true=1500.0,
                                       sound_speed_assumed=1530.0)
    assert z[0] == pytest.approx(d_true * (1530.0 / 1500.0), rel=1e-9)
    # 角度は影響を受けない
    _, th, ph = forward_observation(relative_vector(P, P_PARENT))
    assert z[1] == pytest.approx(th, rel=1e-9)
    assert z[2] == pytest.approx(ph, rel=1e-9)


# --- 8.(C)3  系統バイアス --------------------------------------------------
def test_distance_bias_shifts_mean():
    d_true = np.linalg.norm(P)
    bias_d = 0.5
    N = 4000
    ds = np.array([
        simulate_observation_realistic(P, SIGMA, seed=s, p_parent=P_PARENT,
                                        bias=(bias_d, 0.0, 0.0))[0]
        for s in range(N)])
    # 平均は d_true + bias に寄る (標準誤差 σ/√N の数倍以内)
    se = SIGMA[0] / np.sqrt(N)
    assert abs(ds.mean() - (d_true + bias_d)) < 6 * se


# --- 8.(C)4  距離依存σ -----------------------------------------------------
def test_effective_sigma_grows_linearly():
    s0 = effective_sigma(0.0, SIGMA, range_growth_per_m=0.05)
    s10 = effective_sigma(10.0, SIGMA, range_growth_per_m=0.05)
    assert np.allclose(s0, SIGMA)                       # d=0 で元のσ
    assert s10[1] == pytest.approx(SIGMA[1] * 1.5, rel=1e-12)  # (1 + 0.05*10)
    assert s10[2] == pytest.approx(SIGMA[2] * 1.5, rel=1e-12)
    # 距離成分は dist_growth で別に成長
    sd = effective_sigma(10.0, SIGMA, dist_growth_per_m=0.1)
    assert sd[0] == pytest.approx(SIGMA[0] * 2.0, rel=1e-12)


def test_range_dependent_noise_increases_spread():
    """距離依存ノイズを入れると、遠い点ほど距離成分のばらつきが大きい。"""
    near = np.array([1.8, 0.0, -0.8])     # d ~ 2 m
    far = np.array([0.0, 0.0, -20.0])     # d = 20 m
    k = 0.1
    N = 2000
    sn = np.std([simulate_observation_realistic(near, SIGMA, seed=s, p_parent=P_PARENT,
                                                dist_growth_per_m=k)[0] for s in range(N)])
    sf = np.std([simulate_observation_realistic(far, SIGMA, seed=s, p_parent=P_PARENT,
                                                dist_growth_per_m=k)[0] for s in range(N)])
    assert sf > sn * 2.0                  # d≈20 と d≈2 で σ_d が ~ (1+2)/(1+0.2) ≈ 2.5 倍


# --- 8.(C)5  外れ値率 ------------------------------------------------------
def test_outlier_rate_matches():
    """outlier_rate=0.3 のとき、距離成分が大きく跳ねる割合がほぼ 0.3。"""
    rate = 0.3
    N = 5000
    thresh = 5 * SIGMA[0]                  # 通常ノイズではほぼ超えない閾値
    d_true = np.linalg.norm(P)
    hits = 0
    for s in range(N):
        z = simulate_observation_realistic(P, SIGMA, seed=s, p_parent=P_PARENT,
                                           outlier_rate=rate, outlier_scale=20.0)
        if abs(z[0] - d_true) > thresh:
            hits += 1
    frac = hits / N
    assert abs(frac - rate) < 0.05        # 統計的に rate 近傍


# --- 8.(C)6  時刻同期 (ノイズ0) --------------------------------------------
def test_latency_uses_lagged_position_for_distance():
    zero = (0.0, 0.0, 0.0)
    vel = np.array([1.0, 0.0, 0.0])       # 1 m/s で東進
    dt = 0.2                               # 200 ms 遅延
    p_lag = P - vel * dt
    d_lag = np.linalg.norm(p_lag)
    z = simulate_observation_realistic(P, zero, seed=0, p_parent=P_PARENT,
                                       velocity=vel, acoustic_latency_s=dt)
    assert z[0] == pytest.approx(d_lag, rel=1e-9)
    # 角度は現在位置 P のまま
    _, th, ph = forward_observation(relative_vector(P, P_PARENT))
    assert z[1] == pytest.approx(th, rel=1e-9)
    assert z[2] == pytest.approx(ph, rel=1e-9)
