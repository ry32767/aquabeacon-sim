"""test_attitude.py — MATH_SPEC §14 (親機姿勢と IMU 姿勢推定) の数値テストケース。

各テストは docs/MATH_SPEC.md §14 の (C) 数値テストケースに 1:1 対応する。
  §14.0 Euler<->行列・回転規約
  §14.1 波姿勢真値
  §14.2 機体フレーム観測 (R=I 一致・距離不変)
  §14.3 IMU 生信号 (重力->roll/pitch, 磁気->yaw)
  §14.4 SO(3) 相補フィルタ (静止厳密・動揺厳密・収束・ドリフト)
  §14.5 姿勢補正つき位置推定 (恒等・ワールド復元・端から端まで・後方互換)
"""
import numpy as np
import pytest

from src.attitude import (euler_to_matrix, matrix_to_euler, exp_so3, log_so3,
                          roll_pitch_from_accel, yaw_from_mag,
                          attitude_from_accel_mag, complementary_filter,
                          euler_sequence, body_bearing_to_world,
                          correct_observation_sequence)
from src.sensors import (forward_observation, relative_vector,
                         simulate_observation, simulate_observation_attitude,
                         simulate_observation_sequence_attitude,
                         simulate_observation_sequence, simulate_imu_signals,
                         apply_attitude_error)
from src.truth import wave_attitude_sequence, double_lawnmower_trajectory
from src.estimator import estimate_trajectory
from src.evaluation import rmse_xyz

GRAV = 9.80665
MAG_REF = np.array([0.0, 1.0, 0.0])


# ============================================================ §14.0 規約
def test_identity_at_zero():
    assert np.allclose(euler_to_matrix(0, 0, 0), np.eye(3), atol=1e-12)


def test_yaw_rotates_x_axis():
    psi = 0.6
    got = euler_to_matrix(0, 0, psi) @ np.array([1.0, 0, 0])
    assert np.allclose(got, [np.cos(psi), np.sin(psi), 0.0], atol=1e-9)


def test_pitch_rotates_x_axis():
    th = 0.4
    got = euler_to_matrix(0, th, 0) @ np.array([1.0, 0, 0])
    assert np.allclose(got, [np.cos(th), 0.0, -np.sin(th)], atol=1e-9)


def test_roll_rotates_y_axis():
    ph = 0.5
    got = euler_to_matrix(ph, 0, 0) @ np.array([0.0, 1, 0])
    assert np.allclose(got, [0.0, np.cos(ph), np.sin(ph)], atol=1e-9)


@pytest.mark.parametrize("e", [(0.1, -0.2, 0.3), (-0.3, 0.15, -1.2), (0.05, 0.4, 2.0)])
def test_euler_matrix_roundtrip(e):
    got = matrix_to_euler(euler_to_matrix(*e))
    assert np.allclose(got, e, atol=1e-9)


def test_exp_log_roundtrip():
    rv = np.array([0.2, -0.5, 0.3])
    assert np.allclose(log_so3(exp_so3(rv)), rv, atol=1e-9)
    assert np.allclose(exp_so3(np.zeros(3)), np.eye(3), atol=1e-12)


# ============================================================ §14.1 波姿勢真値
def test_wave_shape():
    e = wave_attitude_sequence(50, dt=0.02, seed=1)
    assert e.shape == (50, 3)


def test_wave_bounded():
    # 各軸 |angle - mean| <= 合成振幅。yaw_mean=0 のケースで上界を確認。
    amp = np.deg2rad(5.0)
    e = wave_attitude_sequence(400, dt=0.02, seed=3,
                               roll_amp=amp, pitch_amp=amp, yaw_amp=amp,
                               yaw_mean=0.0)
    assert np.all(np.abs(e) <= amp + 1e-9)


def test_wave_reproducible():
    a = wave_attitude_sequence(30, dt=0.02, seed=7)
    b = wave_attitude_sequence(30, dt=0.02, seed=7)
    c = wave_attitude_sequence(30, dt=0.02, seed=8)
    assert np.allclose(a, b)
    assert not np.allclose(a, c)


# ============================================================ §14.2 機体観測
def test_body_obs_identity_matches_world():
    # R=I なら機体観測 == §1 simulate_observation (同 seed, ノイズ込みで完全一致)。
    p = np.array([6.0, 8.0, -7.5])
    sigma = (0.03, np.deg2rad(0.3), np.deg2rad(0.3))
    z_body = simulate_observation_attitude(p, np.eye(3), sigma, seed=5)
    z_plain = simulate_observation(p, sigma, seed=5)
    assert np.allclose(z_body, z_plain, atol=1e-12)
    # body_bearing_to_world(z, I) == z (補正なし=恒等)
    assert np.allclose(body_bearing_to_world(z_body, np.eye(3)), z_body, atol=1e-12)


def test_body_obs_distance_invariant_under_rotation():
    p = np.array([6.0, 8.0, -7.5])
    R = euler_to_matrix(0.1, -0.2, 0.5)
    z = simulate_observation_attitude(p, R, (0.0, 0.0, 0.0), seed=0)
    assert np.isclose(z[0], np.linalg.norm(relative_vector(p)), rtol=1e-9)


# ============================================================ §14.3 IMU 信号
def test_accel_gravity_reference():
    R = euler_to_matrix(0.3, -0.2, 0.7)
    acc = R.T @ np.array([0.0, 0.0, GRAV])
    roll, pitch = roll_pitch_from_accel(acc)
    assert np.isclose(roll, 0.3, atol=1e-9)
    assert np.isclose(pitch, -0.2, atol=1e-9)


def test_mag_yaw_reference():
    phi, th, psi = 0.3, -0.2, 0.7
    R = euler_to_matrix(phi, th, psi)
    mag = R.T @ MAG_REF
    assert np.isclose(yaw_from_mag(mag, phi, th), psi, atol=1e-9)


def test_attitude_from_accel_mag_recovers_truth():
    phi, th, psi = -0.25, 0.18, -1.1
    R = euler_to_matrix(phi, th, psi)
    acc = R.T @ np.array([0.0, 0.0, GRAV])
    mag = R.T @ MAG_REF
    assert np.allclose(attitude_from_accel_mag(acc, mag), [phi, th, psi], atol=1e-9)


def test_imu_signals_noisefree_consistency():
    e = wave_attitude_sequence(20, dt=0.02, seed=2)
    R_seq = np.array([euler_to_matrix(*ek) for ek in e])
    sig = simulate_imu_signals(R_seq, dt=0.02, seed=0, gravity=GRAV)
    assert sig["gyro"].shape == (19, 3)
    assert sig["acc"].shape == (20, 3)
    assert sig["mag"].shape == (20, 3)
    # 各時刻の acc/mag から姿勢を直に復元できる (ノイズ0)
    for k in range(20):
        assert np.allclose(attitude_from_accel_mag(sig["acc"][k], sig["mag"][k]),
                           e[k], atol=1e-9)


# ============================================================ §14.4 相補フィルタ
def test_filter_static_exact():
    # 姿勢一定・ノイズ0 -> R_est == R_true 全時刻。
    R = euler_to_matrix(0.2, -0.15, 0.6)
    R_seq = np.tile(R, (15, 1, 1))
    sig = simulate_imu_signals(R_seq, dt=0.02, seed=0, gravity=GRAV)
    est = complementary_filter(sig["gyro"], sig["acc"], sig["mag"], dt=0.02, alpha=0.98)
    assert np.allclose(est, R_seq, atol=1e-9)


def test_filter_moving_exact_noisefree():
    # 動揺・ノイズ0 -> 予測も測定も真値なので R_est == R_true。
    e = wave_attitude_sequence(40, dt=0.02, seed=4)
    R_seq = np.array([euler_to_matrix(*ek) for ek in e])
    sig = simulate_imu_signals(R_seq, dt=0.02, seed=0, gravity=GRAV)
    est = complementary_filter(sig["gyro"], sig["acc"], sig["mag"], dt=0.02, alpha=0.98)
    assert np.allclose(euler_sequence(est), e, atol=1e-6)


def test_filter_converges_from_wrong_init():
    R = euler_to_matrix(0.2, -0.15, 0.6)
    R_seq = np.tile(R, (60, 1, 1))
    sig = simulate_imu_signals(R_seq, dt=0.02, seed=0, gravity=GRAV)
    R0_wrong = euler_to_matrix(0.0, 0.0, 0.0)
    est = complementary_filter(sig["gyro"], sig["acc"], sig["mag"], dt=0.02,
                               alpha=0.9, R0=R0_wrong)
    # 終盤で真値へ収束
    assert np.allclose(est[-1], R, atol=1e-3)


def test_filter_gyro_only_drifts_with_bias():
    # alpha=1 (ジャイロのみ) + バイアスありで姿勢誤差が増大する。
    R = euler_to_matrix(0.1, 0.0, 0.0)
    R_seq = np.tile(R, (200, 1, 1))
    sig = simulate_imu_signals(R_seq, dt=0.02, seed=0, gravity=GRAV,
                               gyro_bias=np.deg2rad(2.0))
    est = complementary_filter(sig["gyro"], sig["acc"], sig["mag"], dt=0.02, alpha=1.0)
    err_early = np.linalg.norm(log_so3(R_seq[5].T @ est[5]))
    err_late = np.linalg.norm(log_so3(R_seq[-1].T @ est[-1]))
    assert err_late > err_early + 1e-3


# ============================================================ §14.5 補正つき位置推定
def test_correction_identity():
    z = np.array([12.5, 0.9, -0.6])
    assert np.allclose(body_bearing_to_world(z, np.eye(3)), z, atol=1e-12)


def test_correction_recovers_world_bearing():
    v_w = relative_vector(np.array([6.0, 8.0, -7.5]))
    R = euler_to_matrix(0.2, -0.3, 0.9)
    z_body = forward_observation(R.T @ v_w)
    z_world = body_bearing_to_world(z_body, R)
    z_true = forward_observation(v_w)
    assert np.allclose(z_world[1:], z_true[1:], atol=1e-9)   # 方位/仰角が一致
    assert np.isclose(z_world[0], z_true[0], rtol=1e-9)      # 距離も一致


def test_end_to_end_noisefree_recovers_trajectory():
    # 動揺する親機の機体観測を、完全 IMU の相補フィルタ姿勢で補正 -> 真の軌道に一致。
    traj = double_lawnmower_trajectory()
    n = len(traj)
    e = wave_attitude_sequence(n, dt=0.02, seed=11)
    R_seq = np.array([euler_to_matrix(*ek) for ek in e])
    sigma = (0.0, 0.0, 0.0)
    z_body = simulate_observation_sequence_attitude(traj, R_seq, sigma, seed=0)
    sig = simulate_imu_signals(R_seq, dt=0.02, seed=0, gravity=GRAV)
    R_est = complementary_filter(sig["gyro"], sig["acc"], sig["mag"],
                                 dt=0.02, alpha=0.98)
    z_world = correct_observation_sequence(z_body, R_est)
    sigma_est = (0.03, np.deg2rad(0.3), np.deg2rad(0.3))   # 推定の重みは公称σ
    est = estimate_trajectory(z_world, sigma_est)
    assert rmse_xyz(traj, est)["total"] < 1e-6


def test_backward_compat_identity_attitude():
    # R_seq=I の補正は観測を変えない -> 姿勢なし (§5) と一致。
    traj = double_lawnmower_trajectory()
    n = len(traj)
    sigma = (0.03, np.deg2rad(0.3), np.deg2rad(0.3))
    R_I = np.tile(np.eye(3), (n, 1, 1))
    z_body = simulate_observation_sequence_attitude(traj, R_I, sigma, seed=0)
    z_plain = simulate_observation_sequence(traj, sigma, seed=0)
    # 角度は同 seed で一致 (距離は機体観測でノイズ0化されるため別途比較)
    assert np.allclose(z_body[:, 1:], z_plain[:, 1:], atol=1e-12)
    z_world = correct_observation_sequence(z_body, R_I)
    assert np.allclose(z_world, z_body, atol=1e-12)


# ----------------------------------------------------------------------------
# 波動揺を「観測誤差」として重ねる合成ヘルパ (apply_attitude_error, §8/§14)
# ----------------------------------------------------------------------------
def test_attitude_error_disabled_is_identity():
    # enable=False なら観測をそのまま返す (全シナリオの既定 = 従来と完全一致)。
    traj = double_lawnmower_trajectory()
    sigma = (0.03, np.deg2rad(0.3), np.deg2rad(0.3))
    z = simulate_observation_sequence(traj, sigma, seed=0)
    assert np.allclose(apply_attitude_error(z, seed=0, enable=False), z, atol=0.0)


def test_attitude_error_single_observation_shape():
    # 単一観測 (3,) を入れたら (3,) で返る (静止点シナリオ用、n=1 で補正も落ちない)。
    z = np.array([10.0, 0.3, -0.9])
    out_naive = apply_attitude_error(z, seed=1, enable=True, imu_correct=False)
    out_corr = apply_attitude_error(z, seed=1, enable=True, imu_correct=True)
    assert out_naive.shape == (3,) and out_corr.shape == (3,)
    assert np.isclose(out_naive[0], z[0]) and np.isclose(out_corr[0], z[0])  # 距離は回転不変


def test_attitude_error_naive_degrades_corrected_recovers():
    # 波動揺ありで naive は劣化し、IMU 補正で baseline 付近へ回復する (§14 の効果)。
    traj = double_lawnmower_trajectory()
    sigma = (0.03, np.deg2rad(0.3), np.deg2rad(0.3))
    wave = {"roll_amp": np.deg2rad(5.0), "pitch_amp": np.deg2rad(4.0),
            "yaw_amp": np.deg2rad(3.0)}
    imu = {"gyro_sigma": np.deg2rad(0.1), "acc_sigma": 0.05, "mag_sigma": 0.02}
    z = simulate_observation_sequence(traj, sigma, seed=0)
    z_naive = apply_attitude_error(z, seed=0, enable=True, imu_correct=False,
                                   wave=wave, dt=0.02)
    z_corr = apply_attitude_error(z, seed=0, enable=True, imu_correct=True,
                                  wave=wave, dt=0.02, filter_alpha=0.98, **imu)
    rb = rmse_xyz(traj, estimate_trajectory(z, sigma))["total"]
    rn = rmse_xyz(traj, estimate_trajectory(z_naive, sigma))["total"]
    rc = rmse_xyz(traj, estimate_trajectory(z_corr, sigma))["total"]
    assert rn > rb                 # naive は baseline より悪化
    assert rc < rn                 # 補正は naive を改善
    assert rc < 3.0 * rb           # baseline の数倍以内まで回復
