"""
test_covariance.py — 解の不確かさ: 解析ヤコビアン・CRLB・GDOP・一貫性 (MATH_SPEC §4.3, §4.5, §15)。

研究グレードの検証:
  - 解析ヤコビアン (§4.3) が数値微分と一致する。
  - 共分散 (JᵀWJ)⁻¹ (§4.5) が経験モンテカルロ共分散と一致する (推定が効率的=CRLBを達成)。
  - NEES (正規化推定誤差二乗) の平均が自由度 3 になる (共分散が正しく較正されている, §15)。
  - 推定がほぼ不偏 (|bias| ≪ RMSE)。
  - 深度融合 (§10) が z 方向の CRLB を縮める。
  - 軌道/SBL の return_cov が形状どおりで、過決定系の共分散が経験値に整合する。
MBD: 共分散は位置と σ のみから決まり truth を見ない (estimator 層, test_separation で担保)。
"""
import numpy as np
import pytest

from src.estimator import (observation_jacobian, _numeric_obs_jacobian,
                           position_covariance, gdop, estimate_trajectory,
                           estimate_trajectory_sbl)
from src import evaluation as ev

SIGMA = (0.03, np.deg2rad(0.3), np.deg2rad(0.3))


# ---- §4.3 解析ヤコビアン ----
@pytest.mark.parametrize("x", [(6, 8, -7.5), (3, 4, 0), (0.5, 0.5, -10),
                               (2, -3, -4), (0.1, 0.0, -12)])
def test_analytic_jacobian_matches_numeric(x):
    Ja = observation_jacobian(np.array(x, float))
    Jn = _numeric_obs_jacobian(np.array(x, float), np.zeros(3), True)
    assert np.allclose(Ja, Jn, atol=1e-6), f"x={x}\nanalytic={Ja}\nnumeric={Jn}"


def test_analytic_jacobian_inverted_frame_sign():
    """observe_from_parent=False で ∂v/∂x=-I の符号反転が正しい。"""
    x = np.array([3, 4, -5.], float)
    Ja = observation_jacobian(x, observe_from_parent=False)
    Jn = _numeric_obs_jacobian(x, np.zeros(3), False)
    assert np.allclose(Ja, Jn, atol=1e-6)


# ---- §4.5 共分散・GDOP ----
def test_crlb_matches_montecarlo_efficiency():
    """単時刻 (観測3・未知3, 厳密決定系) は WLS≈ML なので RMSE が CRLB を達成する。"""
    truth = np.array([6, 8, -7.5], float)
    crlb = ev.crlb_rmse(truth, SIGMA)
    est = ev.monte_carlo_estimates(truth, SIGMA, n=4000, seed=0)
    rmse = ev.rmse_with_ci(est, truth, seed=0)["rmse"]
    assert 0.9 <= rmse / crlb <= 1.15, f"RMSE/CRLB={rmse/crlb:.3f} (rmse={rmse}, crlb={crlb})"


def test_gdop_increases_with_range():
    """同方向で遠いほど GDOP (位置不確かさ) が増える (角度誤差×距離)。"""
    near = gdop(position_covariance(np.array([3, 4, 0], float), SIGMA))
    far = gdop(position_covariance(np.array([9, 12, 0], float), SIGMA))   # 3倍遠い
    assert far > near


def test_depth_fusion_reduces_z_covariance():
    """深度センサ (§10) を足すと z の分散が縮む (鉛直を直接拘束)。"""
    truth = np.array([0.5, 0.5, -12.0], float)        # near-nadir で仰角が z に効きにくい
    cov_no = position_covariance(truth, SIGMA)
    cov_dz = position_covariance(truth, SIGMA, with_depth=True, sigma_depth=0.05)
    assert cov_dz[2, 2] < cov_no[2, 2]


def test_nees_consistency_mean_near_dof():
    """NEES の平均が自由度 3 に近い = 共分散が正しく較正されている (§15)。"""
    truth = np.array([6, 8, -7.5], float)
    est = ev.monte_carlo_estimates(truth, SIGMA, n=6000, seed=1)
    cov = ev.crlb_position(truth, SIGMA)
    ne = ev.nees(truth, est, cov)
    assert 2.7 <= ne.mean() <= 3.3, f"mean NEES={ne.mean():.3f} (dof=3)"


def test_estimator_is_unbiased():
    """推定バイアスが RMSE に対し十分小さい (分散支配, §15 VAL-02)。"""
    truth = np.array([6, 8, -7.5], float)
    est = ev.monte_carlo_estimates(truth, SIGMA, n=8000, seed=2)
    bias = ev.position_bias(truth, est)
    se = est.std(axis=0) / np.sqrt(len(est))          # 軸別の経験標準誤差
    assert np.all(np.abs(bias) < 5.0 * se), f"bias={bias}, se={se}"


def test_rmse_ci_point_matches_rmse_xyz():
    """rmse_with_ci の点推定が従来の rmse_xyz['total'] と一致する (後方互換)。"""
    truth = np.array([2, -3, -4], float)
    est = ev.monte_carlo_estimates(truth, SIGMA, n=1500, seed=3)
    ci = ev.rmse_with_ci(est, truth, seed=0)
    assert np.isclose(ci["rmse"], ev.rmse_xyz(truth, est)["total"], atol=1e-12)
    assert ci["ci_low"] <= ci["rmse"] <= ci["ci_high"]


# ---- 軌道・SBL の共分散 (過決定系) ----
def test_trajectory_return_cov_shape_and_psd():
    """estimate_trajectory(return_cov=True) が (n,3,3) の半正定値共分散を返す。"""
    from src.truth import double_lawnmower_trajectory
    from src.sensors import (simulate_observation_sequence,
                             simulate_imu_displacements)
    traj = double_lawnmower_trajectory(area=(2.0, 1.5), depth=-8.0, n_legs=2,
                                       pts_per_leg=5, origin=(0.5, 0.5))
    z = simulate_observation_sequence(traj, SIGMA, seed=0)
    imu = simulate_imu_displacements(traj, 0.02, seed=100)
    X, cov = estimate_trajectory(z, SIGMA, imu_deltas=imu, sigma_imu=0.02,
                                 return_cov=True)
    assert cov.shape == (len(traj), 3, 3)
    for C in cov:
        assert np.allclose(C, C.T, atol=1e-12)        # 対称
        assert np.all(np.linalg.eigvalsh(C) > -1e-12)  # 半正定値


def test_trajectory_cov_default_unchanged():
    """return_cov=False (既定) は従来どおり (n,3) のみを返す (後方互換)。"""
    from src.truth import double_lawnmower_trajectory
    from src.sensors import simulate_observation_sequence, simulate_imu_displacements
    traj = double_lawnmower_trajectory(area=(2.0, 1.5), depth=-8.0, n_legs=2,
                                       pts_per_leg=5, origin=(0.5, 0.5))
    z = simulate_observation_sequence(traj, SIGMA, seed=0)
    imu = simulate_imu_displacements(traj, 0.02, seed=100)
    out = estimate_trajectory(z, SIGMA, imu_deltas=imu, sigma_imu=0.02)
    assert isinstance(out, np.ndarray) and out.shape == (len(traj), 3)


def test_montecarlo_trajectory_stats_independent_and_ci():
    """montecarlo_trajectory_stats が独立試行で平均RMSE+CI を返す (§15)。"""
    from src.rng import substream_seed
    truth = np.zeros((4, 3))

    def trial_fn(trial_seed):                 # 真値=0, 推定=独立ノイズ N(0,0.1)
        rng = np.random.default_rng(substream_seed(trial_seed, 9))
        return truth, truth + rng.normal(0.0, 0.1, truth.shape)

    st = ev.montecarlo_trajectory_stats(trial_fn, n_trials=50, base_seed=0)
    assert {"mean", "std", "se", "ci_low", "ci_high", "n", "per_trial"} <= set(st)
    assert st["n"] == 50 and len(st["per_trial"]) == 50
    assert st["ci_low"] <= st["mean"] <= st["ci_high"]
    assert 0.1 < st["mean"] < 0.25            # 期待 ~ sqrt(3)*0.1 = 0.173


def test_heteroscedastic_sigma_constant_matches_scalar():
    """(n,3) で全時刻同一σを渡すと (3,) スカラと同一結果 (EST-04 後方互換)。"""
    from src.truth import double_lawnmower_trajectory
    from src.sensors import simulate_observation_sequence, simulate_imu_displacements
    traj = double_lawnmower_trajectory(area=(2.0, 1.5), depth=-8.0, n_legs=2,
                                       pts_per_leg=5, origin=(0.5, 0.5))
    z = simulate_observation_sequence(traj, SIGMA, seed=0)
    imu = simulate_imu_displacements(traj, 0.02, seed=100)
    a = estimate_trajectory(z, SIGMA, imu_deltas=imu, sigma_imu=0.02)
    sig_arr = np.tile(np.array(SIGMA), (len(traj), 1))
    b = estimate_trajectory(z, sig_arr, imu_deltas=imu, sigma_imu=0.02)
    assert np.allclose(a, b, atol=1e-12)
