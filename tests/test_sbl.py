"""test_sbl.py — SBL 親機4トランスデューサ音響測位 (MATH_SPEC §13) のテスト。

4点(既知配置)への距離 → 多辺測量で光学なしに3D測位。単時刻でも可観測。
同一平面アレイは深い子機で z が弱く、深度センサが締めることを検証する。
"""
import numpy as np

from src.config import (SBL_ANCHORS, SBL_SIGMA_RANGE, SIGMA, SIGMA_IMU,
                        SIGMA_DEPTH, P_PARENT)
from src.truth import double_lawnmower_trajectory
from src.sensors import (simulate_sbl_ranges, simulate_sbl_range_sequence,
                         simulate_observation_sequence, simulate_imu_displacements,
                         simulate_depth_sequence, sbl_attitude_anchors)
from src.estimator import (estimate_trajectory_sbl, estimate_trajectory_acoustic_inertial)
from src.evaluation import rmse_xyz


def _traj():
    return double_lawnmower_trajectory(area=(6.0, 4.0), depth=-13.0,
                                       n_legs=2, pts_per_leg=6, origin=(3.0, 3.0))


# --- 観測モデル -----------------------------------------------------------
def test_sbl_ranges_true_distances():
    p = np.array([3.0, 2.0, -13.0])
    d = simulate_sbl_ranges(p, SBL_ANCHORS, 0.0, seed=0)   # ノイズ0
    expected = np.linalg.norm(SBL_ANCHORS - p, axis=1)
    assert np.allclose(d, expected, atol=1e-12)
    assert len(d) == len(SBL_ANCHORS) == 4


def test_sbl_ranges_realistic_backward_compat():
    # §13.4 (C5): 音響誤差を既定 (理想) にすると従来と完全一致 (後方互換)。
    p = np.array([3.0, 2.0, -13.0])
    base = simulate_sbl_ranges(p, SBL_ANCHORS, SBL_SIGMA_RANGE, seed=7)
    same = simulate_sbl_ranges(p, SBL_ANCHORS, SBL_SIGMA_RANGE, seed=7,
                               sound_speed_true=1500.0, sound_speed_assumed=1500.0,
                               bias_dist=0.0, dist_growth_per_m=0.0, outlier_rate=0.0)
    assert np.allclose(base, same, atol=1e-12)


def test_sbl_ranges_sound_speed_scale():
    # §13.4 (C6): 音速ズレ c_assumed/c_true = 1.02 -> 各距離が 1.02*d_true (ノイズ0)。
    p = np.array([3.0, 2.0, -13.0])
    d_true = np.linalg.norm(SBL_ANCHORS - p, axis=1)
    d = simulate_sbl_ranges(p, SBL_ANCHORS, 0.0, seed=0,
                            sound_speed_true=1500.0, sound_speed_assumed=1530.0)
    assert np.allclose(d, d_true * (1530.0 / 1500.0), rtol=1e-9)


# --- 13.(C)1  ノイズフリー可観測性 -----------------------------------------
def test_noise_free_recovery():
    traj = _traj()
    rng = np.array([[np.linalg.norm(a - p) for a in SBL_ANCHORS] for p in traj])
    imu = np.diff(traj, axis=0)
    dep = -traj[:, 2]
    est = estimate_trajectory_sbl(rng, SBL_ANCHORS, SBL_SIGMA_RANGE, imu, SIGMA_IMU,
                                  dep, SIGMA_DEPTH, p_parent=P_PARENT)
    assert rmse_xyz(traj, est)["total"] < 1e-3


# --- 13.(C)2  単時刻可観測 (4距離のみ, IMU/深度なし) ------------------------
def test_observable_ranges_only():
    """4距離だけ (IMUも深度も無し) で位置が定まる (方位不要)。"""
    traj = _traj()
    rng = np.array([[np.linalg.norm(a - p) for a in SBL_ANCHORS] for p in traj])
    est = estimate_trajectory_sbl(rng, SBL_ANCHORS, SBL_SIGMA_RANGE,
                                  None, None, None, None, p_parent=P_PARENT)
    assert rmse_xyz(traj, est)["total"] < 1e-3      # 多辺測量で正確に復元 (z は下解で初期化)


# --- 13.(C)3  深度の寄与 ---------------------------------------------------
def test_depth_helps_vertical():
    """同一平面アレイ+深い子機では z が弱い。深度ありの z RMSE が小さい。"""
    traj = _traj()
    rng = simulate_sbl_range_sequence(traj, SBL_ANCHORS, SBL_SIGMA_RANGE, seed=0)
    imu = simulate_imu_displacements(traj, SIGMA_IMU, seed=1)
    dep = simulate_depth_sequence(traj, SIGMA_DEPTH, seed=2)
    z_no = rmse_xyz(traj, estimate_trajectory_sbl(
        rng, SBL_ANCHORS, SBL_SIGMA_RANGE, imu, SIGMA_IMU, None, None,
        p_parent=P_PARENT))["z"]
    z_dp = rmse_xyz(traj, estimate_trajectory_sbl(
        rng, SBL_ANCHORS, SBL_SIGMA_RANGE, imu, SIGMA_IMU, dep, SIGMA_DEPTH,
        p_parent=P_PARENT))["z"]
    assert z_dp < z_no
    assert z_dp < 0.05                               # 深度で z は ~σ_depth に締まる


# --- 13.(C)4  対 単一距離フォールバック ------------------------------------
def test_sbl_beats_single_range():
    traj = _traj()
    rng = simulate_sbl_range_sequence(traj, SBL_ANCHORS, SBL_SIGMA_RANGE, seed=0)
    z = simulate_observation_sequence(traj, SIGMA, seed=0, p_parent=P_PARENT)
    imu = simulate_imu_displacements(traj, SIGMA_IMU, seed=1)
    dep = simulate_depth_sequence(traj, SIGMA_DEPTH, seed=2)
    sbl = rmse_xyz(traj, estimate_trajectory_sbl(
        rng, SBL_ANCHORS, SBL_SIGMA_RANGE, imu, SIGMA_IMU, dep, SIGMA_DEPTH,
        p_parent=P_PARENT))["total"]
    single = rmse_xyz(traj, estimate_trajectory_acoustic_inertial(
        z[:, 0], SIGMA[0], imu, SIGMA_IMU, dep, SIGMA_DEPTH,
        p_parent=P_PARENT))["total"]
    assert sbl < single                              # 4距離で水平が直接定まり高精度


def test_larger_baseline_improves():
    """アレイ一辺が広いほど (GDOP 改善) RMSE が小さい。"""
    def anchors(B):
        b = B / 2
        return np.array([[b, b, 0.], [b, -b, 0.], [-b, b, 0.], [-b, -b, 0.]])

    traj = _traj()
    imu = simulate_imu_displacements(traj, SIGMA_IMU, seed=1)
    dep = simulate_depth_sequence(traj, SIGMA_DEPTH, seed=2)

    def rmse(B):
        rng = simulate_sbl_range_sequence(traj, anchors(B), SBL_SIGMA_RANGE, seed=0)
        return rmse_xyz(traj, estimate_trajectory_sbl(
            rng, anchors(B), SBL_SIGMA_RANGE, imu, SIGMA_IMU, dep, SIGMA_DEPTH,
            p_parent=P_PARENT))["total"]

    assert rmse(8.0) < rmse(1.0)


# --- 13.5  親機の波動揺による SBL アンカーアレイ回転 ------------------------
def _anchors(B):
    b = B / 2
    return np.array([[b, b, 0.], [b, -b, 0.], [-b, b, 0.], [-b, -b, 0.]])


def _wave():
    # 動揺パラメータを明示 (config 非依存にする)。数 deg の roll/pitch/yaw。
    return dict(roll_amp=np.deg2rad(6.0), pitch_amp=np.deg2rad(5.0),
                yaw_amp=np.deg2rad(8.0), roll_period=4.0, pitch_period=5.0,
                yaw_period=6.0, yaw_mean=0.0)


def test_sbl_attitude_backward_compat():
    # §13.5 (C7): enable=False なら公称アンカーをそのまま返す (真値・推定とも)。
    a_true, a_est = sbl_attitude_anchors(SBL_ANCHORS, n=10, seed=0, enable=False)
    assert np.array_equal(a_true, SBL_ANCHORS)
    assert np.array_equal(a_est, SBL_ANCHORS)


def test_sbl_attitude_rotates_anchors():
    # enable=True で真値アンカーは時刻ごとに動き (n,M,3)、公称から有意にずれる。
    anchors = _anchors(8.0)
    a_true, _ = sbl_attitude_anchors(anchors, n=12, seed=0, enable=True,
                                     imu_correct=False, wave=_wave(), p_parent=P_PARENT)
    assert a_true.shape == (12, len(anchors), 3)
    # ピボットからの距離 (アレイ半径) は回転で不変だが、各アンカー位置はずれる。
    assert np.linalg.norm(a_true[0] - anchors) > 0.05
    assert np.allclose(np.linalg.norm(a_true[0], axis=1),
                       np.linalg.norm(anchors, axis=1), atol=1e-9)


def test_sbl_attitude_worsens_and_imu_corrects():
    # §13.5 (C8): 波動揺ありの naive(補正なし) は波なしより悪化し、IMU 補正は naive より改善。
    traj = _traj()
    n = len(traj)
    anchors = _anchors(8.0)
    imu = simulate_imu_displacements(traj, SIGMA_IMU, seed=1)
    dep = simulate_depth_sequence(traj, SIGMA_DEPTH, seed=2)

    def sbl_rmse(a_true, a_est):
        rng = simulate_sbl_range_sequence(traj, a_true, SBL_SIGMA_RANGE, seed=0)
        est = estimate_trajectory_sbl(rng, a_est, SBL_SIGMA_RANGE, imu, SIGMA_IMU,
                                      dep, SIGMA_DEPTH, p_parent=P_PARENT, loss="huber")
        return rmse_xyz(traj, est)["total"]

    base = sbl_rmse(anchors, anchors)                       # 波動揺なし (基準)
    a_true, a_naive = sbl_attitude_anchors(anchors, n, seed=0, enable=True,
                                           imu_correct=False, wave=_wave(),
                                           p_parent=P_PARENT)
    naive = sbl_rmse(a_true, a_naive)                       # 真値=回転, 推定=公称
    # 低ノイズ IMU なら R_est≈R_true でアンカーをほぼ正しく回せる。
    a_true2, a_corr = sbl_attitude_anchors(anchors, n, seed=0, enable=True,
                                           imu_correct=True, wave=_wave(),
                                           p_parent=P_PARENT, gyro_sigma=1e-3,
                                           acc_sigma=1e-3, mag_sigma=1e-3)
    corr = sbl_rmse(a_true2, a_corr)                        # 真値=回転, 推定=IMU補正
    assert np.array_equal(a_true, a_true2)                  # 真値は seed/wave のみ依存で同一
    assert naive > base                                     # 波動揺で系統誤差が乗る
    assert corr < naive                                     # IMU 補正で改善
