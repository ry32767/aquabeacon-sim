"""evaluation.py — ⑤ 評価 (MATH_SPEC §6 の指標は Stage 2)。

ここでだけ真値と推定値を突き合わせる層 (MBD)。Stage 1 では位置 RMSE を計算する。

- position_error : 推定 - 真値
- rmse_xyz       : 多試行の RMSE (X/Y/Z 別と合成)
- monte_carlo_rmse: ノイズ付き観測→推定を N 回回して RMSE を出す (感度解析の核)
"""
import numpy as np

from src.sensors import simulate_observation
from src.estimator import estimate_position


def position_error(truth, estimate):
    """推定誤差ベクトル estimate - truth [m]。"""
    return np.asarray(estimate, dtype=float) - np.asarray(truth, dtype=float)


def rmse_xyz(truth, estimates):
    """真値と推定群の RMSE を返す。

    truth     : (3,) 真位置 (全試行共通) または (N,3)
    estimates : (3,) 単一推定 または (N,3) 多試行
    戻り値 dict: {'x','y','z','total'} [m]
        per-axis = sqrt(mean(err_axis^2)), total = sqrt(mean(sum(err^2, axis=1)))
    """
    truth = np.asarray(truth, dtype=float)
    estimates = np.asarray(estimates, dtype=float)
    err = estimates - truth
    err = np.atleast_2d(err)            # (N,3)
    per_axis = np.sqrt(np.mean(err**2, axis=0))
    total = np.sqrt(np.mean((err**2).sum(axis=1)))
    return {
        'x': float(per_axis[0]),
        'y': float(per_axis[1]),
        'z': float(per_axis[2]),
        'total': float(total),
    }


def monte_carlo_estimates(truth, sigma, n=2000, seed=0,
                          p_parent=None, observe_from_parent=True):
    """真値 truth に対し、ノイズ付き観測→推定を N 回回した推定群 (n,3) を返す。

    各試行で seed を変えて (seed+i) 独立なノイズを与える。再現性のため seed を固定。
    推定 (estimate_position) には truth を渡さず観測のみを入力する (MBD)。
    可視化 (3D 推定クラウド) や RMSE 計算の共通土台。
    """
    truth = np.asarray(truth, dtype=float)
    estimates = np.empty((n, 3))
    for i in range(n):
        z = simulate_observation(truth, sigma, seed=seed + i,
                                 p_parent=p_parent,
                                 observe_from_parent=observe_from_parent)
        estimates[i] = estimate_position(z, sigma, p_parent=p_parent,
                                          observe_from_parent=observe_from_parent)
    return estimates


def monte_carlo_rmse(truth, sigma, n=2000, seed=0,
                     p_parent=None, observe_from_parent=True):
    """真値 truth に対し、ノイズ付き観測→推定を N 回回して RMSE [m] を返す。

    戻り値: rmse_xyz と同じ dict ('x','y','z','total') [m]。
    注意: ここは評価層なので truth を使ってよい。
    """
    estimates = monte_carlo_estimates(truth, sigma, n=n, seed=seed,
                                      p_parent=p_parent,
                                      observe_from_parent=observe_from_parent)
    return rmse_xyz(truth, estimates)


# ----------------------------------------------------------------------------
# Stage 2: ジオメトリ評価 (MATH_SPEC §6)。真値 (寸法・体積・表面) と突き合わせる。
# ----------------------------------------------------------------------------
def dimension_error_mm(L_hat, L_true):
    """寸法誤差 = (L_hat - L_true) を mm で返す (MATH_SPEC §6.1)。"""
    return float((L_hat - L_true) * 1000.0)


def volume_error_rate_pct(V_hat, V_true):
    """体積誤差率 = (V_hat - V_true)/V_true * 100 [%] (MATH_SPEC §6.1)。"""
    return float((V_hat - V_true) / V_true * 100.0)


def pointcloud_rms_to_surface(est_points, true_surface_points):
    """点群距離 RMS = sqrt(mean_i(min_dist(p_est_i, surface_true)^2)) [m] (MATH_SPEC §6.1)。

    各推定点について真の表面点群への最近傍距離を取り、その二乗平均平方根を返す。
    ここは評価層なので真の表面 (truth) を参照してよい。
    """
    from scipy.spatial import cKDTree
    est = np.asarray(est_points, dtype=float).reshape(-1, 3)
    surf = np.asarray(true_surface_points, dtype=float).reshape(-1, 3)
    dist, _ = cKDTree(surf).query(est, k=1)
    return float(np.sqrt(np.mean(dist**2)))
