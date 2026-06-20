"""estimator.py — ③ 位置推定 (MATH_SPEC §3, §4)。

真値を見てはならない層。入力は観測値 (d, theta, phi) と既知の親機位置・ノイズ
パラメータのみ。重み付き最小二乗で子機位置を推定する。

このモジュールは truth を import しない (MBD 層分離。test_separation が強制)。
観測モデル h は sensors の純粋な変換関数 (forward/inverse_observation) を使う。
"""
import numpy as np
from scipy.optimize import least_squares

from src.sensors import forward_observation, inverse_observation, relative_vector


def wrap_angle(a):
    """角度差を (-pi, pi] に正規化する (MATH_SPEC §3.2)。±pi 境界の不連続を防ぐ。"""
    return np.arctan2(np.sin(a), np.cos(a))


def h(x_state, p_parent=None, observe_from_parent=True):
    """観測モデル: 推定状態 x=(x,y,z) -> 予測観測 (d_hat, theta_hat, phi_hat)  (MATH_SPEC §3.1)。"""
    if p_parent is None:
        p_parent = np.zeros(3)
    v = relative_vector(x_state, p_parent, observe_from_parent)
    return forward_observation(v)


def residual(x_state, z_meas, p_parent=None, observe_from_parent=True):
    """残差 r(x) = z - h(x)。角度成分は wrap_angle で正規化する (MATH_SPEC §3.2)。"""
    d_hat, th_hat, ph_hat = h(x_state, p_parent, observe_from_parent)
    d, th, ph = z_meas
    return np.array([
        d - d_hat,
        wrap_angle(th - th_hat),
        wrap_angle(ph - ph_hat),
    ])


def weight_matrix(sigma_dist, sigma_az, sigma_el):
    """重み行列 W = diag(1/sigma^2)  (MATH_SPEC §4.1)。精度の良い観測ほど重い。"""
    return np.diag([1.0 / sigma_dist**2, 1.0 / sigma_az**2, 1.0 / sigma_el**2])


def estimate_position(z_meas, sigma, p_parent=None,
                      observe_from_parent=True, x0=None):
    """重み付き最小二乗で子機位置を推定する (MATH_SPEC §4.2)。

    z_meas : 観測 (d, theta, phi)
    sigma  : (sigma_dist [m], sigma_az [rad], sigma_el [rad])
    p_parent: 親機位置 (既知)。None なら原点。
    x0     : 初期値。None なら §2 の逆変換で算出する。
    戻り値 : 推定位置 x_hat=(x,y,z)

    least_squares は残差ベクトルを最小化するので、重み W=diag(1/sigma^2) の平方根
    sqrt(W)=diag(1/sigma) を残差に掛けて渡す (r^T W r と等価)。
    """
    if p_parent is None:
        p_parent = np.zeros(3)
    sd, sa, se = sigma
    sqrtW = np.array([1.0 / sd, 1.0 / sa, 1.0 / se])

    if x0 is None:
        v0 = inverse_observation(*z_meas)
        x0 = (p_parent + v0) if observe_from_parent else (p_parent - v0)

    def weighted_residual(x_state):
        return sqrtW * residual(x_state, z_meas, p_parent, observe_from_parent)

    sol = least_squares(weighted_residual, x0, method='lm')
    return sol.x


def estimate_trajectory(z_seq, sigma_obs, imu_deltas=None, sigma_imu=None,
                        p_parent=None, observe_from_parent=True, x0=None):
    """複数時刻の観測 (+ IMU 拘束) から子機軌道を一括推定する (MATH_SPEC §5)。

    z_seq     : (n,3) 各時刻の観測 (d, theta, phi)
    sigma_obs : (sigma_dist, sigma_az, sigma_el) 観測ノイズ
    imu_deltas: (n-1,3) 時刻間変位 delta_p の IMU 観測。None なら IMU 拘束なし。
    sigma_imu : IMU 変位ノイズ (スカラ or (3,))。imu_deltas を使うとき必須。
    x0        : 初期軌道 (n,3)。None なら各時刻を §2 逆変換で初期化する。
    戻り値    : 推定軌道 X_hat (n,3)

    目的関数 (MATH_SPEC §5.3):
        X_hat = argmin_X  Σ_k ||r_obs_k||^2_{W_obs} + Σ_k ||r_imu_k||^2_{W_imu}
        r_obs_k = z_k - h(x_k)              (3節)
        r_imu_k = (x_{k+1} - x_k) - delta_p_imu_k
    least_squares に全残差を連結したベクトルを渡して最小化する。
    入力は観測値のみ。truth は一切参照しない (MBD)。
    """
    if p_parent is None:
        p_parent = np.zeros(3)
    z_seq = np.asarray(z_seq, dtype=float)
    n = len(z_seq)

    sd, sa, se = sigma_obs
    sqrtW_obs = np.array([1.0 / sd, 1.0 / sa, 1.0 / se])

    use_imu = imu_deltas is not None
    if use_imu:
        if sigma_imu is None:
            raise ValueError("imu_deltas を使うときは sigma_imu が必要です")
        imu_deltas = np.asarray(imu_deltas, dtype=float)
        sqrtW_imu = 1.0 / np.broadcast_to(np.asarray(sigma_imu, dtype=float), (3,))

    # 初期値: 各時刻を逆変換で
    if x0 is None:
        x0 = np.empty((n, 3))
        for k in range(n):
            v0 = inverse_observation(*z_seq[k])
            x0[k] = (p_parent + v0) if observe_from_parent else (p_parent - v0)
    x0 = np.asarray(x0, dtype=float).reshape(n, 3)

    def stacked_residual(x_flat):
        X = x_flat.reshape(n, 3)
        parts = []
        for k in range(n):       # 観測残差
            parts.append(sqrtW_obs * residual(X[k], z_seq[k], p_parent,
                                              observe_from_parent))
        if use_imu:              # IMU 拘束残差
            for k in range(n - 1):
                parts.append(sqrtW_imu * ((X[k + 1] - X[k]) - imu_deltas[k]))
        return np.concatenate(parts)

    sol = least_squares(stacked_residual, x0.ravel(), method='lm')
    return sol.x.reshape(n, 3)
