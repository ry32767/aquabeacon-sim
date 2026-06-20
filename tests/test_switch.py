"""test_switch.py — 光学↔フォールバック自動切替 (MATH_SPEC §12) のテスト。

見失い率+ヒステリシスの状態機械で角度マスクを作り、光学が健全な時刻は角度を、
見失い多発の時刻は距離+IMU+深度のみを使う。光学ブラックアウト下でも軌道を保つ。
"""
import numpy as np
import pytest

from src.config import SIGMA, SIGMA_IMU, SIGMA_DEPTH, P_PARENT
from src.truth import double_lawnmower_trajectory
from src.sensors import (simulate_observation_sequence, simulate_imu_displacements,
                         simulate_depth_sequence)
from src.estimator import (estimate_trajectory, estimate_trajectory_auto,
                           estimate_trajectory_acoustic_inertial, optical_health_mask)
from src.evaluation import rmse_xyz


def _traj():
    return double_lawnmower_trajectory(area=(6.0, 4.0), depth=-10.0,
                                       n_legs=3, pts_per_leg=8, origin=(3.0, 3.0))


# --- 12.(C)1  状態機械の基本 ----------------------------------------------
def test_mask_all_detected():
    det = np.ones(20, bool)
    assert optical_health_mask(det).all()


def test_mask_none_detected():
    det = np.zeros(20, bool)
    assert not optical_health_mask(det).any()


def test_mask_blackout_window():
    """中央のブラックアウトで mask が False に落ち、回復後に True に戻る (ヒステリシス遅れ)。"""
    det = np.array([True] * 10 + [False] * 10 + [True] * 10)
    mask = optical_health_mask(det, threshold=0.2, hysteresis=0.05, window=5)
    assert mask[3]                     # 序盤は光学
    assert not mask[14]                # ブラックアウト中はフォールバック
    assert not mask[12]                # 未検出フレームは当然 False
    assert mask[28]                    # 回復後は光学に復帰
    # 回復は即時でなく数フレーム遅れる (窓+ヒステリシス)
    assert not mask[20]


# --- 12.(C)2  自動切替が両モードのいいとこ取り ------------------------------
def _blackout_case(seed=0):
    traj = _traj()
    n = len(traj)
    z = simulate_observation_sequence(traj, SIGMA, seed=seed, p_parent=P_PARENT)
    imu = simulate_imu_displacements(traj, SIGMA_IMU, seed=seed + 1)
    dep = simulate_depth_sequence(traj, SIGMA_DEPTH, seed=seed + 2)
    det = np.ones(n, bool)
    det[int(n * 0.35):int(n * 0.75)] = False          # 中央をブラックアウト
    # 見失いフレームの角度は誤検出 (大外れ値)
    z_bad = z.copy()
    rng = np.random.default_rng(seed + 9)
    for k in range(n):
        if not det[k]:
            z_bad[k, 1] += rng.uniform(-0.5, 0.5)
            z_bad[k, 2] += rng.uniform(-0.5, 0.5)
    return traj, z, z_bad, imu, dep, det


def test_auto_beats_naive_optical_and_fallback():
    traj, z, z_bad, imu, dep, det = _blackout_case(seed=0)
    naive = estimate_trajectory(z_bad, SIGMA, imu_deltas=imu, sigma_imu=SIGMA_IMU,
                                p_parent=P_PARENT, loss="huber")
    fb = estimate_trajectory_acoustic_inertial(z[:, 0], SIGMA[0], imu, SIGMA_IMU,
                                               dep, SIGMA_DEPTH, p_parent=P_PARENT)
    auto, mask = estimate_trajectory_auto(z_bad, SIGMA, det, imu, SIGMA_IMU,
                                          dep, SIGMA_DEPTH, p_parent=P_PARENT)
    r_naive = rmse_xyz(traj, naive)["total"]
    r_fb = rmse_xyz(traj, fb)["total"]
    r_auto = rmse_xyz(traj, auto)["total"]
    assert r_auto <= r_fb + 0.01            # フォールバック以上に良い
    assert r_auto < 0.5 * r_naive           # 素朴な光学維持より大幅に良い
    assert r_auto < 0.1                      # 10 cm 以内
    assert 0 < mask.sum() < len(traj)        # 一部時刻だけ光学を使った (混在)


# --- 12.(C)3  端ケース: 全検出=光学相当 / 全消失=フォールバック相当 --------
def test_all_detected_uses_optical():
    traj = _traj()
    n = len(traj)
    z = simulate_observation_sequence(traj, SIGMA, seed=3, p_parent=P_PARENT)
    imu = simulate_imu_displacements(traj, SIGMA_IMU, seed=4)
    dep = simulate_depth_sequence(traj, SIGMA_DEPTH, seed=5)
    auto, mask = estimate_trajectory_auto(z, SIGMA, np.ones(n, bool), imu, SIGMA_IMU,
                                          dep, SIGMA_DEPTH, p_parent=P_PARENT)
    assert mask.all()
    full_opt = estimate_trajectory(z, SIGMA, imu_deltas=imu, sigma_imu=SIGMA_IMU,
                                   p_parent=P_PARENT, loss="huber",
                                   z_depth_seq=dep, sigma_depth=SIGMA_DEPTH)
    assert rmse_xyz(traj, auto)["total"] == pytest.approx(
        rmse_xyz(traj, full_opt)["total"], abs=5e-3)


def test_all_lost_is_fallback():
    traj = _traj()
    n = len(traj)
    z = simulate_observation_sequence(traj, SIGMA, seed=6, p_parent=P_PARENT)
    imu = simulate_imu_displacements(traj, SIGMA_IMU, seed=7)
    dep = simulate_depth_sequence(traj, SIGMA_DEPTH, seed=8)
    auto, mask = estimate_trajectory_auto(z, SIGMA, np.zeros(n, bool), imu, SIGMA_IMU,
                                          dep, SIGMA_DEPTH, p_parent=P_PARENT)
    assert not mask.any()
    assert rmse_xyz(traj, auto)["total"] < 0.15    # 純フォールバック相当で実用域
