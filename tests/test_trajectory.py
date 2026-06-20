"""
test_trajectory.py — MATH_SPEC §5 (複数時刻 + IMU 拘束) のテスト (Stage 2)。

- ノイズフリー一致: ノイズ0なら複数時刻推定が真の軌道に一致する (IMU 有無とも)。
- IMU 拘束の効果: ノイズ込みで、IMU 拘束を加えると軌道 RMSE が下がる。
- MBD: estimate_trajectory は truth を渡さず観測のみを入力する。

estimator は truth を import しない (test_separation が別途強制)。
"""
import numpy as np

from src.config import SIGMA, SIGMA_IMU, P_PARENT
from src.truth import double_lawnmower_trajectory
from src.sensors import (forward_observation, simulate_observation_sequence,
                         simulate_imu_displacements)
from src.estimator import estimate_trajectory
from src.evaluation import rmse_xyz


def _small_traj():
    # 小さめのダブル芝刈り軌道 (真上 rho=0 を通らない)
    return double_lawnmower_trajectory(area=(6.0, 4.0), depth=-7.5,
                                       n_legs=2, pts_per_leg=4, origin=(3.0, 4.0))


def _forward_seq(traj):
    return np.array([forward_observation(p) for p in traj])


def test_trajectory_noise_free_no_imu():
    """ノイズ0・IMUなしで真の軌道に一致する (atol=1e-6)。"""
    traj = _small_traj()
    z = _forward_seq(traj)
    est = estimate_trajectory(z, SIGMA, p_parent=P_PARENT)
    assert np.allclose(est, traj, atol=1e-6), f"max err={np.abs(est-traj).max()}"


def test_trajectory_noise_free_with_imu():
    """ノイズ0・IMU(真の変位)拘束付きでも真の軌道に一致する (atol=1e-6)。"""
    traj = _small_traj()
    z = _forward_seq(traj)
    imu = np.diff(traj, axis=0)                  # ノイズフリーの真変位
    est = estimate_trajectory(z, SIGMA, imu_deltas=imu, sigma_imu=SIGMA_IMU,
                              p_parent=P_PARENT)
    assert np.allclose(est, traj, atol=1e-6), f"max err={np.abs(est-traj).max()}"


def test_imu_constraint_reduces_rmse():
    """ノイズ込みで、IMU 拘束を加えると軌道 RMSE が下がる (複数 seed の平均で評価)。"""
    traj = _small_traj()
    rmse_no, rmse_imu = [], []
    for seed in range(10):
        z = simulate_observation_sequence(traj, SIGMA, seed=1000 * seed,
                                          p_parent=P_PARENT)
        imu = simulate_imu_displacements(traj, SIGMA_IMU, seed=7 + seed)
        est_no = estimate_trajectory(z, SIGMA, p_parent=P_PARENT)
        est_imu = estimate_trajectory(z, SIGMA, imu_deltas=imu,
                                      sigma_imu=SIGMA_IMU, p_parent=P_PARENT)
        rmse_no.append(rmse_xyz(traj, est_no)["total"])
        rmse_imu.append(rmse_xyz(traj, est_imu)["total"])
    mean_no = float(np.mean(rmse_no))
    mean_imu = float(np.mean(rmse_imu))
    assert mean_imu < mean_no, f"IMU無 {mean_no*1000:.1f}mm <= IMU有 {mean_imu*1000:.1f}mm"


def test_trajectory_estimate_shape():
    """戻り値が入力軌道と同じ (n,3) 形状になる。"""
    traj = _small_traj()
    z = _forward_seq(traj)
    est = estimate_trajectory(z, SIGMA, p_parent=P_PARENT)
    assert est.shape == traj.shape
