"""test_depth.py — 深度センサ融合 (MATH_SPEC §10) の数値テスト。

圧力センサで鉛直 z を直接拘束する。観測モデル・後方互換・z精度向上・
冗長性による単時刻ロバスト棄却を検証する。
"""
import numpy as np
import pytest

from src.config import SIGMA, P_PARENT
from src.sensors import (forward_observation, simulate_observation, simulate_depth)
from src.estimator import (predicted_depth, depth_residual, estimate_position)
from src.evaluation import rmse_xyz

TRUTH = np.array([6.0, 8.0, -7.5])      # depth = 7.5 m
SIGMA_DEPTH = 0.05


# --- 10.(C)1  観測モデル ---------------------------------------------------
def test_predicted_depth_and_residual():
    assert predicted_depth(TRUTH) == pytest.approx(7.5)
    # ノイズ0の深度観測 = 7.5、真値での残差は 0
    z_depth = -TRUTH[2]
    assert depth_residual(TRUTH, z_depth) == pytest.approx(0.0, abs=1e-12)


def test_simulate_depth_mean():
    N = 5000
    vals = np.array([simulate_depth(TRUTH, SIGMA_DEPTH, seed=s) for s in range(N)])
    assert abs(vals.mean() - 7.5) < 6 * SIGMA_DEPTH / np.sqrt(N)


# --- 10.(C)2  ノイズフリー一致 ---------------------------------------------
def test_noise_free_with_depth():
    z = forward_observation(TRUTH)
    z_depth = -TRUTH[2]
    x_hat = estimate_position(z, SIGMA, p_parent=P_PARENT,
                              z_depth=z_depth, sigma_depth=SIGMA_DEPTH)
    assert np.allclose(x_hat, TRUTH, atol=1e-6)


# --- 10.(C)3  後方互換 -----------------------------------------------------
def test_backward_compat_without_depth():
    z = simulate_observation(TRUTH, SIGMA, seed=3, p_parent=P_PARENT)
    a = estimate_position(z, SIGMA, p_parent=P_PARENT)
    b = estimate_position(z, SIGMA, p_parent=P_PARENT, z_depth=None)
    assert np.allclose(a, b, atol=1e-12)


def test_requires_sigma_depth():
    z = forward_observation(TRUTH)
    with pytest.raises(ValueError):
        estimate_position(z, SIGMA, p_parent=P_PARENT, z_depth=7.5)


# --- 10.(C)4  z 精度向上 ---------------------------------------------------
def test_depth_improves_z_axis():
    """仰角ノイズが大きい条件で、深度ありの z軸RMSE が深度なしより小さい。"""
    sigma_bad = (SIGMA[0], np.deg2rad(0.3), np.deg2rad(3.0))   # 仰角を悪化
    N = 1500
    est_no, est_dp = np.empty((N, 3)), np.empty((N, 3))
    for i in range(N):
        z = simulate_observation(TRUTH, sigma_bad, seed=i, p_parent=P_PARENT)
        zd = simulate_depth(TRUTH, SIGMA_DEPTH, seed=10000 + i)
        est_no[i] = estimate_position(z, sigma_bad, p_parent=P_PARENT)
        est_dp[i] = estimate_position(z, sigma_bad, p_parent=P_PARENT,
                                      z_depth=zd, sigma_depth=SIGMA_DEPTH)
    z_no = rmse_xyz(TRUTH, est_no)["z"]
    z_dp = rmse_xyz(TRUTH, est_dp)["z"]
    assert z_dp < 0.3 * z_no                 # z 軸が大幅改善
    assert z_dp < 0.1                        # 深度σ級 (~5cm) に収まる


# --- 10.(C)5  単時刻ロバスト棄却 (冗長性) ----------------------------------
def test_depth_enables_single_time_robust_rejection():
    """仰角 φ に外れ値を注入。深度あり+huber が深度なし/L2 より真値に近い。"""
    z = forward_observation(TRUTH).copy()
    z[2] += np.deg2rad(12.0)                  # 仰角 φ に外れ値
    zd = -TRUTH[2]                            # 深度はクリーン

    # 深度なし L2 (冗長性ゼロ -> 外れ値をそのまま受ける)
    e_no = estimate_position(z, SIGMA, p_parent=P_PARENT)
    # 深度あり + robust (冗長度1 -> φ外れ値を減衰)
    e_dp = estimate_position(z, SIGMA, p_parent=P_PARENT, loss="huber",
                             z_depth=zd, sigma_depth=SIGMA_DEPTH)
    err_no = np.linalg.norm(e_no - TRUTH)
    err_dp = np.linalg.norm(e_dp - TRUTH)
    assert err_dp < 0.5 * err_no
