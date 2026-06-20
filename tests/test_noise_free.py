"""
test_noise_free.py — ノイズ0なら推定が真値に一致することを検証する (MATH_SPEC §4 (C))。

ミニマム (Stage 1) の完了条件のひとつ。estimator は観測のみを入力に、
ノイズフリーなら真値に一致する (atol=1e-6)。
"""
import numpy as np

from src.sensors import forward_observation, simulate_observation
from src.estimator import estimate_position
from src.evaluation import rmse_xyz

SIGMA = (0.03, np.deg2rad(0.3), np.deg2rad(0.3))


def test_estimate_noise_free_recovers_truth():
    """§4 (C) 1: forward の観測を入れると真値に戻る。"""
    truth = np.array([6, 8, -7.5], float)
    z = forward_observation(truth)
    est = estimate_position(z, SIGMA)
    assert np.allclose(est, truth, atol=1e-6), f"est={est} truth={truth}"


def test_estimate_converges_from_offset_initial():
    """§4 (C) 3: 初期値を真値から 5m ずらしてもノイズフリーなら収束する。"""
    truth = np.array([6, 8, -7.5], float)
    z = forward_observation(truth)
    est = estimate_position(z, SIGMA, x0=truth + np.array([5, -5, 5], float))
    assert np.allclose(est, truth, atol=1e-6), f"est={est} truth={truth}"


def test_simulate_observation_zero_noise_is_forward():
    """sigma=0 のノイズ付き観測は forward そのものになる (sensors の健全性)。"""
    truth = np.array([6, 8, -7.5], float)
    z = simulate_observation(truth, (0.0, 0.0, 0.0), seed=0)
    assert np.allclose(z, forward_observation(truth), atol=1e-12)


def test_minimum_rmse_zero_when_noise_free():
    """ノイズフリーなら RMSE ≈ 0 (Stage 1 ミニマム基準)。"""
    truth = np.array([6, 8, -7.5], float)
    z = forward_observation(truth)
    est = estimate_position(z, SIGMA)
    r = rmse_xyz(truth, est)
    assert r['total'] < 1e-6, f"RMSE total = {r['total']}"


def test_estimate_does_not_need_truth():
    """推定は真値を渡さず観測のみで動く (MBD の確認)。"""
    truth = np.array([2, -3, -4], float)
    z = simulate_observation(truth, SIGMA, seed=42)
    est = estimate_position(z, SIGMA)        # truth は渡していない
    # ノイズ込みでも妥当な近さ (角度誤差×距離のオーダー内)
    assert np.linalg.norm(est - truth) < 0.5, f"est={est} truth={truth}"
