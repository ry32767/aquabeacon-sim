"""test_no_optical.py — 光学なし (距離+IMU+深度) の位置推定 (MATH_SPEC §11) のテスト。

光学追跡が使えない場合のフォールバック。単時刻は方位が不可観測 (距離+深度の2拘束)。
IMU で時刻間を繋ぐと軌道が可観測になることを検証する。
"""
import numpy as np
import pytest

from src.config import SIGMA, SIGMA_IMU, SIGMA_DEPTH, P_PARENT
from src.truth import double_lawnmower_trajectory
from src.sensors import (forward_observation, simulate_observation_sequence,
                         simulate_imu_displacements, simulate_depth_sequence)
from src.estimator import (estimate_trajectory, estimate_trajectory_acoustic_inertial)
from src.evaluation import rmse_xyz


def _traj():
    return double_lawnmower_trajectory(area=(6.0, 4.0), depth=-13.0,
                                       n_legs=2, pts_per_leg=6, origin=(3.0, 3.0))


# --- 11.(C)1  ノイズフリー可観測性 -----------------------------------------
def test_noise_free_recovery():
    traj = _traj()
    rng = np.array([np.linalg.norm(p) for p in traj])     # 真の距離
    imu = np.diff(traj, axis=0)                            # 真の変位
    dep = -traj[:, 2]                                      # 真の深度
    est = estimate_trajectory_acoustic_inertial(rng, SIGMA[0], imu, SIGMA_IMU,
                                                dep, SIGMA_DEPTH, p_parent=P_PARENT)
    assert rmse_xyz(traj, est)["total"] < 1e-3            # mm 未満で一致


# --- 11.(C)2  ノイズ下で実用精度 -------------------------------------------
def test_recovery_under_noise():
    traj = _traj()
    z = simulate_observation_sequence(traj, SIGMA, seed=0, p_parent=P_PARENT)
    rng = z[:, 0]                                          # 距離のみ使う (光学なし)
    imu = simulate_imu_displacements(traj, SIGMA_IMU, seed=1)
    dep = simulate_depth_sequence(traj, SIGMA_DEPTH, seed=2)
    est = estimate_trajectory_acoustic_inertial(rng, SIGMA[0], imu, SIGMA_IMU,
                                                dep, SIGMA_DEPTH, p_parent=P_PARENT)
    r = rmse_xyz(traj, est)
    assert r["total"] < 0.15            # 15 cm 以内
    assert r["z"] < 0.08                # z は深度センサで ~σ_depth に締まる


# --- 11.(C)3  深度が z を締める --------------------------------------------
def test_depth_pins_vertical():
    """水平サーベイ (ほぼ一定深) では距離+IMU だけだと z が緩い。深度で z が締まる。"""
    traj = _traj()
    z = simulate_observation_sequence(traj, SIGMA, seed=5, p_parent=P_PARENT)
    rng = z[:, 0]
    imu = simulate_imu_displacements(traj, SIGMA_IMU, seed=6)
    dep = simulate_depth_sequence(traj, SIGMA_DEPTH, seed=7)
    est = estimate_trajectory_acoustic_inertial(rng, SIGMA[0], imu, SIGMA_IMU,
                                                dep, SIGMA_DEPTH, p_parent=P_PARENT)
    assert rmse_xyz(traj, est)["z"] < 0.08


# --- 11.(C)4  use_angles=False は x0 必須 -----------------------------------
def test_no_angles_requires_x0():
    traj = _traj()
    z = simulate_observation_sequence(traj, SIGMA, seed=0, p_parent=P_PARENT)
    imu = np.diff(traj, axis=0)
    with pytest.raises(ValueError):
        estimate_trajectory(z, SIGMA, imu_deltas=imu, sigma_imu=SIGMA_IMU,
                            p_parent=P_PARENT, use_angles=False)


# --- 11.(C)5  光学なしは光学ありに近い (やや劣る程度) ----------------------
def test_no_optical_close_to_optical():
    traj = _traj()
    z = simulate_observation_sequence(traj, SIGMA, seed=3, p_parent=P_PARENT)
    imu = simulate_imu_displacements(traj, SIGMA_IMU, seed=4)
    dep = simulate_depth_sequence(traj, SIGMA_DEPTH, seed=8)
    est_opt = estimate_trajectory(z, SIGMA, imu_deltas=imu, sigma_imu=SIGMA_IMU,
                                  p_parent=P_PARENT)
    est_no = estimate_trajectory_acoustic_inertial(z[:, 0], SIGMA[0], imu, SIGMA_IMU,
                                                   dep, SIGMA_DEPTH, p_parent=P_PARENT)
    r_opt = rmse_xyz(traj, est_opt)["total"]
    r_no = rmse_xyz(traj, est_no)["total"]
    assert r_no < 4.0 * r_opt + 0.05    # 光学なしでも同オーダー (数倍以内)


# --- §11.2.1 退化運動と鏡像不定性 / 適応多スタート (EST-05) ---
def test_straight_line_warns_mirror_ambiguity():
    """ほぼ直進の水平運動では方位不可観測 -> 警告が出る (MATH_SPEC §11.2.1)。"""
    import warnings
    n = 8
    straight = np.column_stack([np.linspace(0.5, 5.0, n), np.full(n, 0.5),
                                np.full(n, -9.0)])
    z = simulate_observation_sequence(straight, SIGMA, seed=3, p_parent=P_PARENT)
    imu = simulate_imu_displacements(straight, SIGMA_IMU, seed=4)
    dep = simulate_depth_sequence(straight, SIGMA_DEPTH, seed=5)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        estimate_trajectory_acoustic_inertial(z[:, 0], SIGMA[0], imu, SIGMA_IMU,
                                              dep, SIGMA_DEPTH, p_parent=P_PARENT)
    assert any(issubclass(x.category, RuntimeWarning) for x in w), \
        "直進運動で鏡像不定の警告が出ていない"


def test_adaptive_multistart_matches_full_grid_on_rich_motion():
    """方位リッチな芝刈りでは適応(1スタート)が多スタート結果と一致する (EST-05)。"""
    import warnings
    traj = double_lawnmower_trajectory(area=(2.0, 1.5), depth=-9.0, n_legs=2,
                                       pts_per_leg=4, origin=(0.5, 0.5))
    z = simulate_observation_sequence(traj, SIGMA, seed=3, p_parent=P_PARENT)
    imu = simulate_imu_displacements(traj, SIGMA_IMU, seed=4)
    dep = simulate_depth_sequence(traj, SIGMA_DEPTH, seed=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        e_auto = estimate_trajectory_acoustic_inertial(z[:, 0], SIGMA[0], imu,
                                                       SIGMA_IMU, dep, SIGMA_DEPTH,
                                                       p_parent=P_PARENT)
        e_grid = estimate_trajectory_acoustic_inertial(z[:, 0], SIGMA[0], imu,
                                                       SIGMA_IMU, dep, SIGMA_DEPTH,
                                                       p_parent=P_PARENT,
                                                       n_azimuth_starts=12)
    assert np.allclose(e_auto, e_grid, atol=1e-6)
