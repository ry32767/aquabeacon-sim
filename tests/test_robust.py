"""test_robust.py — ロバスト推定 (MATH_SPEC §4.4) の数値テスト。

純L2 (linear) は外れ値に弱い。IMU拘束つき軌道推定で huber/cauchy 損失を使うと、
外れ値の時刻の残差が減衰され、軌道全体の RMSE が改善することを検証する。
既定 loss='linear' は従来挙動と完全一致 (後方互換) も確認する。
"""
import numpy as np
import pytest

from src.config import SIGMA, SIGMA_IMU, P_PARENT
from src.truth import true_child_position, double_lawnmower_trajectory
from src.sensors import (forward_observation, simulate_observation_sequence,
                         simulate_imu_displacements)
from src.estimator import estimate_position, estimate_trajectory
from src.evaluation import rmse_xyz

TRUTH = np.array([6.0, 8.0, -7.5])


# --- 4.4(C)1  後方互換: loss='linear' は既定と一致 --------------------------
def test_linear_is_default_position():
    z = forward_observation(TRUTH) + np.array([0.05, 0.002, -0.003])
    a = estimate_position(z, SIGMA, p_parent=P_PARENT)               # 既定
    b = estimate_position(z, SIGMA, p_parent=P_PARENT, loss="linear")
    assert np.allclose(a, b, atol=1e-9)


def test_invalid_loss_raises():
    z = forward_observation(TRUTH)
    with pytest.raises(ValueError):
        estimate_position(z, SIGMA, p_parent=P_PARENT, loss="nope")


# --- 4.4(C)2  ノイズフリーなら robust でも真値に収束 ------------------------
@pytest.mark.parametrize("loss", ["huber", "cauchy", "soft_l1"])
def test_noise_free_recovery_robust(loss):
    z = forward_observation(TRUTH)                 # ノイズ0
    x_hat = estimate_position(z, SIGMA, p_parent=P_PARENT, loss=loss)
    assert np.allclose(x_hat, TRUTH, atol=1e-6)


def test_noise_free_trajectory_robust():
    traj = double_lawnmower_trajectory(area=(6.0, 4.0), depth=-7.5,
                                       n_legs=2, pts_per_leg=5, origin=(3.0, 4.0))
    z = np.array([forward_observation(p) for p in traj])   # ノイズ0
    imu = np.diff(traj, axis=0)
    est = estimate_trajectory(z, SIGMA, imu_deltas=imu, sigma_imu=SIGMA_IMU,
                              p_parent=P_PARENT, loss="huber")
    assert np.allclose(est, traj, atol=1e-5)


# --- 4.4(C)3  外れ値棄却 (headline) -----------------------------------------
def _trajectory_with_outliers(seed=0):
    """小さめのダブル芝刈り軌道に小ノイズ観測 + 数時刻の大外れ値を作る。"""
    traj = double_lawnmower_trajectory(area=(6.0, 4.0), depth=-7.5,
                                       n_legs=2, pts_per_leg=5, origin=(3.0, 4.0))
    z = simulate_observation_sequence(traj, SIGMA, seed=seed, p_parent=P_PARENT)
    # 2時刻に大外れ値を注入 (ライト見失い/音響マルチパス相当)
    z[3, 0] += 3.0                       # 距離 +3 m
    z[3, 1] += np.deg2rad(18.0)          # 方位 +18 deg
    z[7, 0] -= 2.5                       # 距離 -2.5 m
    z[7, 2] += np.deg2rad(15.0)          # 仰角 +15 deg
    imu = simulate_imu_displacements(traj, SIGMA_IMU, seed=seed + 1)
    return traj, z, imu


def test_robust_beats_linear_under_outliers():
    traj, z, imu = _trajectory_with_outliers(seed=0)
    est_lin = estimate_trajectory(z, SIGMA, imu_deltas=imu, sigma_imu=SIGMA_IMU,
                                  p_parent=P_PARENT, loss="linear")
    est_rob = estimate_trajectory(z, SIGMA, imu_deltas=imu, sigma_imu=SIGMA_IMU,
                                  p_parent=P_PARENT, loss="cauchy", f_scale=1.345)
    r_lin = rmse_xyz(traj, est_lin)["total"]
    r_rob = rmse_xyz(traj, est_rob)["total"]
    # ロバストが明確に改善 (少なくとも 40% 減)
    assert r_rob < 0.6 * r_lin


def test_robust_matches_linear_without_outliers():
    """外れ値が無ければ、robust と linear はほぼ同じ (内れ値を壊さない)。"""
    traj = double_lawnmower_trajectory(area=(6.0, 4.0), depth=-7.5,
                                       n_legs=2, pts_per_leg=5, origin=(3.0, 4.0))
    z = simulate_observation_sequence(traj, SIGMA, seed=5, p_parent=P_PARENT)
    imu = simulate_imu_displacements(traj, SIGMA_IMU, seed=6)
    est_lin = estimate_trajectory(z, SIGMA, imu_deltas=imu, sigma_imu=SIGMA_IMU,
                                  p_parent=P_PARENT, loss="linear")
    est_rob = estimate_trajectory(z, SIGMA, imu_deltas=imu, sigma_imu=SIGMA_IMU,
                                  p_parent=P_PARENT, loss="huber", f_scale=1.345)
    r_lin = rmse_xyz(traj, est_lin)["total"]
    r_rob = rmse_xyz(traj, est_rob)["total"]
    assert abs(r_rob - r_lin) < 0.3 * r_lin       # 大きくは変わらない
