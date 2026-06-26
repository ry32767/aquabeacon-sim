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


# ============================================================================
# §8.4b SVP (音速プロファイル実効音速) / §8.6 時間相関 / §5.5 IMU バイアス
# いずれも既定 (g=0, rho=0, sigma_bias=0) で従来と完全一致 (後方互換) を担保する。
# ============================================================================
from src.sensors import (effective_sound_speed, gauss_markov_sequence,
                         simulate_sbl_range_sequence, simulate_depth_sequence,
                         simulate_imu_displacements)


# --- 8.4b SVP --------------------------------------------------------------
def test_svp_gradient_zero_is_constant_speed():
    """g=0 なら実効音速は c0 (従来の定数音速と一致)。"""
    assert effective_sound_speed(1500.0, 0.0, -15.0) == 1500.0


def test_svp_effective_speed_is_log_mean():
    """g!=0 の実効音速は端点速度の対数平均 (c_child-c0)/ln(c_child/c0)。"""
    c0, g, z = 1500.0, -0.5, -20.0
    c_child = c0 + g * z
    expect = (c_child - c0) / np.log(c_child / c0)
    assert effective_sound_speed(c0, g, z) == pytest.approx(expect, rel=1e-12)


def test_svp_backward_compatible_observation():
    """svp_gradient_per_s=0 (既定) は観測を変えない (後方互換)。"""
    for seed in range(10):
        a = simulate_observation_realistic(P, SIGMA, seed=seed, p_parent=P_PARENT,
                                           sound_speed_true=1505.0)
        b = simulate_observation_realistic(P, SIGMA, seed=seed, p_parent=P_PARENT,
                                           sound_speed_true=1505.0,
                                           svp_gradient_per_s=0.0)
        assert np.allclose(a, b, atol=1e-12)


def test_svp_shifts_range_when_enabled():
    """g!=0 でノイズ0なら測距が実効音速スケールで変わる (定量)。"""
    p = np.array([0.5, 0.5, -18.0])
    zero = (0.0, 0.0, 0.0)
    d_true = np.linalg.norm(p)
    c0 = 1500.0
    z = simulate_observation_realistic(p, zero, seed=0, sound_speed_true=c0,
                                       sound_speed_assumed=c0, svp_gradient_per_s=-0.4)
    c_eff = effective_sound_speed(c0, -0.4, -18.0)
    assert z[0] == pytest.approx(d_true * c0 / c_eff, rel=1e-9)


# --- 8.6 時間相関ノイズ (AR(1)) --------------------------------------------
def test_colored_noise_rho0_matches_white_sequence():
    """rho=0 の観測列は従来の白色 (per-step seed) と完全一致 (後方互換)。"""
    traj = np.array([[3.0, 4.0, -5.0], [3.2, 4.1, -5.1], [3.4, 4.2, -5.2],
                     [3.6, 4.0, -5.0]])
    a = simulate_observation_sequence(traj, SIGMA, seed=11)
    b = simulate_observation_sequence(traj, SIGMA, seed=11, rho=0.0)
    assert np.array_equal(a, b)


def test_gauss_markov_marginal_variance_and_autocorr():
    """定常 AR(1): 周辺 std ≈ sigma, lag-1 自己相関 ≈ rho (統計, 固定 seed)。"""
    e = gauss_markov_sequence(40000, 0.5, 0.7, seed=1)
    assert abs(e.std() - 0.5) < 0.02
    ac = np.corrcoef(e[:-1], e[1:])[0, 1]
    assert abs(ac - 0.7) < 0.03


def test_colored_depth_and_sbl_rho0_backward_compatible():
    """深度/SBL 列も rho=0 で従来と完全一致。"""
    traj = np.array([[0.5, 0.5, -8.0], [0.7, 0.6, -8.2], [0.9, 0.7, -8.1]])
    anchors = np.array([[1., 1., 0.], [1., -1., 0.], [-1., 1., 0.], [-1., -1., 0.]])
    assert np.array_equal(simulate_depth_sequence(traj, 0.05, seed=5),
                          simulate_depth_sequence(traj, 0.05, seed=5, rho=0.0))
    assert np.array_equal(simulate_sbl_range_sequence(traj, anchors, 0.03, seed=5),
                          simulate_sbl_range_sequence(traj, anchors, 0.03, seed=5, rho=0.0))


# --- 5.5 IMU 変位バイアス ---------------------------------------------------
def test_imu_bias_zero_backward_compatible():
    """sigma_bias=0, bias0=0 (既定) で従来の白色出力と完全一致。"""
    traj = np.array([[0.0, 0.0, -5.0], [0.3, 0.1, -5.0], [0.6, 0.2, -5.0],
                     [0.9, 0.1, -5.0]])
    a = simulate_imu_displacements(traj, 0.02, seed=4)
    b = simulate_imu_displacements(traj, 0.02, seed=4, sigma_bias=0.0, bias0=0.0)
    assert np.array_equal(a, b)


def test_imu_bias_adds_drift():
    """sigma_bias>0 で変位が白色のみと変わる (バイアスドリフト付与); b_0=bias0。"""
    traj = np.array([[0.0, 0.0, -5.0], [0.3, 0.1, -5.0], [0.6, 0.2, -5.0],
                     [0.9, 0.1, -5.0]])
    a = simulate_imu_displacements(traj, 0.02, seed=4)
    b = simulate_imu_displacements(traj, 0.02, seed=4, sigma_bias=0.01, bias0=0.0)
    assert not np.array_equal(a, b)
    assert np.allclose(a[0], b[0], atol=1e-12)     # b_0=bias0=0 -> 初回は白色のまま
