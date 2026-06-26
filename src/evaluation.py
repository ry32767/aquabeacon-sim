"""evaluation.py — ⑤ 評価 (MATH_SPEC §6 の指標は Stage 2)。

ここでだけ真値と推定値を突き合わせる層 (MBD)。Stage 1 では位置 RMSE を計算する。

- position_error : 推定 - 真値
- rmse_xyz       : 多試行の RMSE (X/Y/Z 別と合成)
- monte_carlo_rmse: ノイズ付き観測→推定を N 回回して RMSE を出す (感度解析の核)
"""
import numpy as np

from src.sensors import simulate_observation
from src.estimator import estimate_position, position_covariance


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
# 解の不確かさの検証 (MATH_SPEC §4.5, §15): CRLB・信頼区間・一貫性 (NEES)・不偏性。
# ここは評価層なので truth を見てよい。CRLB は estimator.position_covariance を
# **真値の幾何で**評価したもの (truth を引数で渡すだけ。estimator は truth を見ない)。
# ----------------------------------------------------------------------------
def crlb_position(truth, sigma, p_parent=None, observe_from_parent=True,
                  with_depth=False, sigma_depth=None):
    """位置推定の Cramér-Rao 下界 (CRLB) 共分散 3x3 [m²] を返す (MATH_SPEC §4.5, §15)。

    真値の観測幾何での (JᵀWJ)⁻¹。効率的推定量 (WLS≈ML) なら経験 RMSE がこれに漸近する。
    """
    return position_covariance(np.asarray(truth, dtype=float), sigma, p_parent,
                               observe_from_parent, with_depth=with_depth,
                               sigma_depth=sigma_depth)


def crlb_rmse(truth, sigma, p_parent=None, observe_from_parent=True,
              with_depth=False, sigma_depth=None):
    """CRLB の合成 RMSE = sqrt(trace(CRLB)) [m] (理論下界, MATH_SPEC §4.5, §15)。"""
    cov = crlb_position(truth, sigma, p_parent, observe_from_parent,
                        with_depth=with_depth, sigma_depth=sigma_depth)
    return float(np.sqrt(np.trace(cov)))


def rmse_with_ci(estimates, truth, level=0.95, n_boot=2000, seed=0):
    """合成 RMSE のブートストラップ信頼区間を返す (MATH_SPEC §15)。

    estimates: (N,3) 推定群, truth: (3,) または (N,3)。
    戻り値 dict: {rmse, se, ci_low, ci_high, n, level} [m]。
    試行を非復元でなく復元再標本化 (bootstrap) し、各リサンプルの RMSE 分布から
    パーセンタイル区間を作る。点推定 (rmse) は従来の rmse_xyz['total'] と一致する。
    再現性のため seed を固定 (np.random.default_rng)。
    """
    est = np.atleast_2d(np.asarray(estimates, dtype=float))
    tr = np.asarray(truth, dtype=float)
    sq = np.sum((est - tr) ** 2, axis=1)            # 各試行の二乗誤差 |err|² (N,)
    N = len(sq)
    point = float(np.sqrt(sq.mean()))
    rng = np.random.default_rng(seed)
    boots = np.sqrt(rng.choice(sq, size=(n_boot, N), replace=True).mean(axis=1))
    alpha = (1.0 - level) / 2.0
    lo, hi = np.percentile(boots, [100 * alpha, 100 * (1.0 - alpha)])
    return {"rmse": point, "se": float(boots.std(ddof=1)),
            "ci_low": float(lo), "ci_high": float(hi), "n": int(N), "level": level}


def nees(truth, estimates, cov):
    """正規化推定誤差二乗 NEES = eᵀ P⁻¹ e の配列を返す (推定一貫性, MATH_SPEC §15)。

    truth: (3,), estimates: (N,3), cov: (3,3) 共通 または (N,3,3) 試行別。
    共分散が正しく較正されていれば NEES の平均は自由度 dof=3 に一致する
    (平均≪3: 共分散が過大/保守的, 平均≫3: 過小/楽観的)。
    """
    err = np.atleast_2d(np.asarray(estimates, dtype=float) - np.asarray(truth, dtype=float))
    cov = np.asarray(cov, dtype=float)
    if cov.ndim == 2:
        P = np.linalg.inv(cov)
        return np.einsum("ni,ij,nj->n", err, P, err)
    out = np.empty(len(err))
    for i in range(len(err)):
        out[i] = float(err[i] @ np.linalg.inv(cov[i]) @ err[i])
    return out


def position_bias(truth, estimates):
    """推定の系統バイアス E[x_hat]-x_true を軸別に返す (3,) [m] (不偏性検査, MATH_SPEC §15)。

    RMSE はバイアスと分散を混ぜるため、バイアス成分を分離して報告する。near-nadir では
    非線形最小二乗に微小バイアスが出るので、分散支配 (|bias|≪RMSE) を確認するのに使う。
    """
    err = np.atleast_2d(np.asarray(estimates, dtype=float) - np.asarray(truth, dtype=float))
    return err.mean(axis=0)


def error_distribution_stats(errors, fail_threshold):
    """誤差列の分布統計を返す (信頼性指標, MATH_SPEC §15)。

    平均 RMSE は稀な発散を平均に埋もれさせる。中央値・95 パーセンタイル・**粗大故障率**
    (誤差 > fail_threshold の割合) を併報すると、稀だが致命的な発散を可視化できる。

    errors        : 各試行の誤差 (同一単位, 例 RMSE [mm] or [m])
    fail_threshold: 粗大故障とみなすしきい値 (errors と同単位)
    戻り値 dict   : {median, p95, mean, fail_rate, n}
    """
    e = np.asarray(errors, dtype=float)
    return {"median": float(np.median(e)), "p95": float(np.percentile(e, 95)),
            "mean": float(e.mean()), "fail_rate": float(np.mean(e > fail_threshold)),
            "n": int(len(e))}


def montecarlo_trajectory_stats(trial_fn, n_trials, base_seed=0, level=0.95,
                                n_boot=2000, boot_seed=0):
    """軌道推定を**独立な乱数で** n_trials 回まわし、平均 RMSE と信頼区間を返す (MATH_SPEC §15)。

    各試行は互いに素なサブストリーム (rng.substream_seed) から生成されるので、従来の
    `base+s` 連番が招く試行間ノイズ再利用 (REP-01, 過小分散) を構造的に防ぐ。各試行の
    軌道 RMSE = sqrt(mean_k ‖x̂_k - x_k‖²) を集め、その平均にブートストラップ CI を付ける。

    trial_fn : (trial_seed:int) -> (truth_traj (n,3), est_traj (n,3))。
               trial_seed を基にセンサ別サブストリームを派生させること
               (例: substream_seed(trial_seed, 0/1/2) を obs/imu/depth に)。
    n_trials : 試行数 (論文の比較曲線は >=30 推奨)。
    戻り値 dict: {mean, std, se, ci_low, ci_high, n, per_trial(list)} [m]。
    """
    from src.rng import substream_seed
    rt = np.empty(n_trials)
    for s in range(n_trials):
        truth, est = trial_fn(substream_seed(base_seed, s))
        err = np.asarray(est, dtype=float) - np.asarray(truth, dtype=float)
        rt[s] = np.sqrt(np.mean(np.sum(err**2, axis=1)))
    mean = float(rt.mean())
    std = float(rt.std(ddof=1)) if n_trials > 1 else 0.0
    se = std / np.sqrt(n_trials) if n_trials > 1 else 0.0
    rng = np.random.default_rng(boot_seed)
    boots = rng.choice(rt, size=(n_boot, n_trials), replace=True).mean(axis=1)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.percentile(boots, [100 * alpha, 100 * (1.0 - alpha)])
    return {"mean": mean, "std": std, "se": float(se), "ci_low": float(lo),
            "ci_high": float(hi), "n": int(n_trials), "per_trial": rt.tolist()}


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
